import argparse
import base64
import html
import io
import uuid
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as F

from fivek_project.model import build_model
from predict import pick_device


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FiveK Photo Enhancer</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f1ec;
      color: #171717;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(135deg, rgba(249, 247, 241, 0.9), rgba(229, 235, 232, 0.92)),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'%3E%3Cpath d='M0 0h120v120H0z' fill='%23f4f1ec'/%3E%3Cpath d='M10 20h100M10 60h100M10 100h100M20 10v100M60 10v100M100 10v100' stroke='%23d9d2c4' stroke-width='1' opacity='.4'/%3E%3C/svg%3E");
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}
    header {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0;
      font-size: 32px;
      line-height: 1.1;
      font-weight: 780;
    }}
    .status {{
      font-size: 14px;
      color: #435048;
    }}
    .panel {{
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid #d8d3c8;
      border-radius: 8px;
      box-shadow: 0 18px 50px rgba(48, 42, 30, 0.09);
      padding: 18px;
    }}
    form {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
    }}
    input[type="file"] {{
      width: 100%;
      min-height: 42px;
      border: 1px solid #bbb5aa;
      border-radius: 6px;
      background: #fff;
      padding: 8px;
      font-size: 15px;
    }}
    button, a.button {{
      border: 0;
      border-radius: 6px;
      background: #1f3d36;
      color: white;
      font-size: 15px;
      font-weight: 700;
      min-height: 42px;
      padding: 0 18px;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }}
    .message {{
      margin: 16px 0 0;
      color: #8a2d21;
      font-weight: 650;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 22px;
    }}
    figure {{
      margin: 0;
      background: #fff;
      border: 1px solid #d8d3c8;
      border-radius: 8px;
      overflow: hidden;
    }}
    figcaption {{
      padding: 10px 12px;
      font-size: 14px;
      font-weight: 750;
      color: #2f332f;
      border-bottom: 1px solid #e3ded4;
    }}
    img {{
      display: block;
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
      background: #ece7de;
    }}
    .actions {{
      margin-top: 14px;
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    @media (max-width: 720px) {{
      header, form, .grid {{
        grid-template-columns: 1fr;
        display: grid;
      }}
      h1 {{
        font-size: 26px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>FiveK Photo Enhancer</h1>
      <div class="status">Checkpoint: {checkpoint}</div>
    </header>
    <section class="panel">
      <form method="post" action="/enhance" enctype="multipart/form-data">
        <input type="file" name="photo" accept="image/*" required>
        <button type="submit">Enhance</button>
      </form>
      {message}
    </section>
    {result}
  </main>
</body>
</html>
"""


RESULT_HTML = """
<section class="grid">
  <figure>
    <figcaption>Original</figcaption>
    <img src="data:image/jpeg;base64,{original}">
  </figure>
  <figure>
    <figcaption>Enhanced</figcaption>
    <img src="data:image/jpeg;base64,{enhanced}">
  </figure>
</section>
<div class="actions">
  <a class="button" href="/download/{filename}">Download enhanced image</a>
</div>
"""


class PhotoEnhancerServer(BaseHTTPRequestHandler):
    model = None
    device = None
    checkpoint = None
    image_size = 256
    output_dir = Path("outputs/web")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_page()
            return
        if parsed.path.startswith("/download/"):
            self.send_download(parsed.path.removeprefix("/download/"))
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/enhance":
            self.send_error(404)
            return

        try:
            filename, image_bytes = self.read_uploaded_image()
            original, enhanced, output_name = self.enhance(filename, image_bytes)
            result = RESULT_HTML.format(
                original=image_to_base64(original),
                enhanced=image_to_base64(enhanced),
                filename=html.escape(output_name),
            )
            self.send_page(result=result)
        except Exception as error:
            self.send_page(message=f"<p class='message'>{html.escape(str(error))}</p>")

    def read_uploaded_image(self) -> tuple[str, bytes]:
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {content_type}\r\n\r\n".encode() + body
        )

        for part in message.iter_parts():
            if part.get_param("name", header="content-disposition") == "photo":
                filename = part.get_filename() or "photo.jpg"
                payload = part.get_payload(decode=True)
                if payload:
                    return filename, payload
        raise ValueError("No image upload found.")

    @torch.no_grad()
    def enhance(self, filename: str, image_bytes: bytes) -> tuple[Image.Image, Image.Image, str]:
        original = Image.open(io.BytesIO(image_bytes))
        original = ImageOps.exif_transpose(original).convert("RGB")
        resized = ImageOps.fit(
            original,
            (self.image_size, self.image_size),
            method=Image.Resampling.LANCZOS,
        )

        tensor = F.to_tensor(resized).unsqueeze(0).to(self.device)
        output = self.model(tensor).squeeze(0).cpu().clamp(0, 1)
        enhanced = F.to_pil_image(output)

        safe_stem = Path(filename).stem.replace(" ", "_") or "photo"
        output_name = f"{safe_stem}_{uuid.uuid4().hex[:8]}_enhanced.jpg"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        enhanced.save(self.output_dir / output_name, quality=95)
        return resized, enhanced, output_name

    def send_page(self, message: str = "", result: str = "") -> None:
        page = HTML_PAGE.format(
            checkpoint=html.escape(str(self.checkpoint)),
            message=message,
            result=result,
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page.encode())))
        self.end_headers()
        self.wfile.write(page.encode())

    def send_download(self, filename: str) -> None:
        path = self.output_dir / Path(filename).name
        if not path.exists():
            self.send_error(404)
            return

        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local web UI for the FiveK enhancer.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train first with train.py.")

    device = pick_device()
    model = build_model(pretrained=False).to(device)
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model"])
    model.eval()

    PhotoEnhancerServer.model = model
    PhotoEnhancerServer.device = device
    PhotoEnhancerServer.checkpoint = checkpoint
    PhotoEnhancerServer.image_size = args.image_size

    server = ThreadingHTTPServer((args.host, args.port), PhotoEnhancerServer)
    print(f"Open http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the web app.")
    server.serve_forever()


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode()


if __name__ == "__main__":
    main()
