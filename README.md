# Harrison and Jerry's Photo Project

Image-to-text photo editing suggestions using the MIT-Adobe FiveK dataset.

The input is the original unedited FiveK photo. The target is structured text
that describes how Expert D changed the image: brightness, contrast, highlights,
shadows, temperature, tint, saturation, and clarity. Since FiveK does not include
natural-language suggestions directly, this project converts the difference
between the original image and Expert D's retouched image into text labels.

Large data folders are intentionally kept out of Git. The repo expects the
source folders to live next to this project folder:

```text
142B final project/
  Harrison-and-Jerry-s-photo-proj/
  raw/
  d/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Prepare the 100-Image Subset

This creates 256 x 256 RGB pairs and an 80/10/10 split:

```bash
python prepare_data.py --raw-dir ../raw --target-dir ../d --limit 100 --image-size 256
```

The generated folder is ignored by Git:

```text
data/
  train/
    input/
    target/
  val/
    input/
    target/
  test/
    input/
    target/
```

## Train

Train the image-to-text suggestion model:

```bash
python train.py --epochs 15 --batch-size 8
```

The model uses pretrained ResNet18 image features and fine-tunes a small
regression head that predicts Expert D-style editing changes.

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/suggestions/best.pt --split test
```

Evaluation reports regression loss and prints example text suggestions.

## Use on New Photos

Generate suggestions for one new image:

```bash
python predict.py --input path/to/photo.jpg --checkpoint checkpoints/suggestions/best.pt
```

Generate suggestions for every image in `new_photos/`:

```bash
python predict.py --input new_photos --checkpoint checkpoints/suggestions/best.pt
```

## Local Web App

Run a browser UI for uploading and enhancing images:

```bash
python web_app.py --checkpoint checkpoints/suggestions/best.pt
```

Then open `http://127.0.0.1:8000`. The page shows the original image,
a manual slider preview, and model-generated text suggestions based on the input
image. The web preview uses the uploaded image at its original resolution, while
the model still analyzes a 256 x 256 copy internally.

## Research Question

Can a neural network learn to analyze an original photo and generate useful
editing suggestions that approximate Expert D's professional retouching style?
