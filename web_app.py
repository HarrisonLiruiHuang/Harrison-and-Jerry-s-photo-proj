import argparse
import base64
import html
import io
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as F

from fivek_project.suggestion_model import build_suggestion_model
from fivek_project.suggestions import (
    slider_defaults_from_labels,
    suggestions_from_labels,
    tensor_to_labels,
)
from predict import pick_device


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FiveK Editing Suggestions</title>
  <style>
    :root {{
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #eef1ed;
      color: #161817;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(135deg, #f7f4ef, #e8efec 48%, #eef0f5);
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 44px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 30px;
      line-height: 1.1;
    }}
    .status {{
      color: #4a534e;
      font-size: 14px;
    }}
    .panel, figure, .suggestions, .controls {{
      background: rgba(255,255,255,.88);
      border: 1px solid #d1d8d2;
      border-radius: 8px;
      box-shadow: 0 16px 42px rgba(33, 43, 37, .08);
    }}
    .panel {{
      padding: 16px;
      margin-bottom: 18px;
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
      border: 1px solid #b8c0ba;
      border-radius: 6px;
      background: white;
      padding: 8px;
      font-size: 15px;
    }}
    button {{
      border: 0;
      border-radius: 6px;
      background: #214236;
      color: white;
      font-size: 15px;
      font-weight: 760;
      min-height: 42px;
      padding: 0 18px;
      cursor: pointer;
    }}
    .message {{
      margin: 12px 0 0;
      color: #8a2d21;
      font-weight: 650;
    }}
    .workspace {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      align-items: start;
    }}
    figure {{
      margin: 0;
      overflow: hidden;
    }}
    figcaption {{
      padding: 10px 12px;
      font-weight: 760;
      border-bottom: 1px solid #dde2de;
    }}
    img {{
      display: block;
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
      background: #e5e5e0;
    }}
    .manual-preview {{
      filter:
        brightness(var(--brightness, 1))
        contrast(var(--contrast, 1))
        saturate(var(--saturation, 1))
        sepia(var(--warmth, 0));
    }}
    .suggestions {{
      margin-top: 16px;
      padding: 16px 18px;
    }}
    .suggestions h2, .controls h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      line-height: 1.2;
    }}
    .suggestions ul {{
      margin: 0;
      padding-left: 20px;
    }}
    .suggestions li {{
      margin: 7px 0;
    }}
    .controls {{
      margin-top: 16px;
      padding: 16px;
      display: grid;
      gap: 13px;
    }}
    .control {{
      display: grid;
      grid-template-columns: 116px 1fr 48px;
      gap: 12px;
      align-items: center;
      font-size: 14px;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: #214236;
    }}
    .value {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: #34413a;
    }}
    @media (max-width: 760px) {{
      header, form, .workspace {{
        display: grid;
        grid-template-columns: 1fr;
      }}
      h1 {{
        font-size: 25px;
      }}
      .control {{
        grid-template-columns: 1fr;
        gap: 5px;
      }}
      .value {{
        text-align: left;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>FiveK Editing Suggestions</h1>
      <div class="status">Image-to-text model: {checkpoint}</div>
    </header>
    <section class="panel">
      <form method="post" action="/suggest" enctype="multipart/form-data">
        <input type="file" name="photo" accept="image/*" required>
        <button type="submit">Get suggestions</button>
      </form>
      {message}
    </section>
    {result}
  </main>
  <script>
    const preview = document.querySelector("[data-preview]");
    const controls = document.querySelectorAll("[data-control]");
    const updatePreview = () => {{
      if (!preview) return;
      const values = {{}};
      controls.forEach((control) => {{
        const name = control.dataset.control;
        const value = Number(control.value);
        values[name] = value;
        const output = document.querySelector(`[data-value="${{name}}"]`);
        if (output) output.textContent = value > 0 ? `+${{value}}` : String(value);
      }});
      preview.style.setProperty("--brightness", String(1 + (values.brightness || 0) / 100));
      preview.style.setProperty("--contrast", String(1 + (values.contrast || 0) / 100));
      preview.style.setProperty("--saturation", String(1 + (values.saturation || 0) / 100));
      preview.style.setProperty("--warmth", String(Math.max(0, values.temperature || 0) / 160));
    }};
    controls.forEach((control) => control.addEventListener("input", updatePreview));
    updatePreview();
  </script>
</body>
</html>
"""


RESULT_HTML = """
<section class="workspace">
  <figure>
    <figcaption>Original input image</figcaption>
    <img src="data:image/jpeg;base64,{image}">
  </figure>
  <figure>
    <figcaption>Manual slider preview</figcaption>
    <img data-preview class="manual-preview" src="data:image/jpeg;base64,{image}">
  </figure>
</section>
<section class="controls">
  <h2>Manual adjustment sliders</h2>
  {controls}
</section>
<section class="suggestions">
  <h2>Model-generated editing suggestions</h2>
  {suggestions}
</section>
"""


CONTROL_HTML = """
<label class="control">
  <span>{label}</span>
  <input data-control="{name}" type="range" min="-60" max="60" value="{value}">
  <span class="value" data-value="{name}">{value}</span>
</label>
"""


class SuggestionServer(BaseHTTPRequestHandler):
    model = None
    device = None
    checkpoint = None
    image_size = 256

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/":
            self.send_page()
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/suggest":
            self.send_error(404)
            return

        try:
            image_bytes = self.read_uploaded_image()
            image, labels, suggestions = self.suggest(image_bytes)
            result = RESULT_HTML.format(
                image=image_to_base64(image),
                controls=controls_to_html(slider_defaults_from_labels(labels)),
                suggestions=suggestions_to_html(suggestions),
            )
            self.send_page(result=result)
        except Exception as error:
            self.send_page(message=f"<p class='message'>{html.escape(str(error))}</p>")

    def read_uploaded_image(self) -> bytes:
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {content_type}\r\n\r\n".encode() + body
        )

        for part in message.iter_parts():
            if part.get_param("name", header="content-disposition") == "photo":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload
        raise ValueError("No image upload found.")

    @torch.no_grad()
    def suggest(self, image_bytes: bytes) -> tuple[Image.Image, dict[str, float], list[str]]:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = ImageOps.fit(image, (self.image_size, self.image_size), method=Image.Resampling.LANCZOS)

        tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        prediction = self.model(tensor).squeeze(0)
        labels = tensor_to_labels(prediction)
        suggestions = suggestions_from_labels(labels, image)
        return image, labels, suggestions

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

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local web UI for image-to-text edit suggestions.")
    parser.add_argument("--checkpoint", default="checkpoints/suggestions/best.pt")
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
    model = build_suggestion_model(pretrained=False).to(device)
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model"])
    model.eval()

    SuggestionServer.model = model
    SuggestionServer.device = device
    SuggestionServer.checkpoint = checkpoint
    SuggestionServer.image_size = args.image_size

    server = ThreadingHTTPServer((args.host, args.port), SuggestionServer)
    print(f"Open http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the web app.")
    server.serve_forever()


def controls_to_html(defaults: dict[str, int]) -> str:
    labels = {
        "brightness": "Brightness",
        "contrast": "Contrast",
        "saturation": "Saturation",
        "temperature": "Warmth",
        "clarity": "Clarity",
    }
    return "".join(
        CONTROL_HTML.format(name=name, label=label, value=defaults.get(name, 0))
        for name, label in labels.items()
    )


def suggestions_to_html(suggestions: list[str]) -> str:
    items = "".join(f"<li>{html.escape(suggestion)}</li>" for suggestion in suggestions)
    return f"<ul>{items}</ul>"


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode()


if __name__ == "__main__":
    main()
