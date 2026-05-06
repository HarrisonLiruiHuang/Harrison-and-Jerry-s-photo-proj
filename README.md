# Harrison and Jerry's Photo Project

Image-to-image photo enhancement using the MIT-Adobe FiveK dataset.

The input image is the original unedited FiveK photo. The target image is the
Expert D retouched version. The model is a simple U-Net decoder fine-tuned on
top of an ImageNet-pretrained ResNet18 encoder.

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

L1-only baseline:

```bash
python train.py --epochs 10 --batch-size 4
```

Optional L1 + SSIM fine-tuning:

```bash
python train.py --epochs 10 --batch-size 4 --ssim-weight 0.1
```

The default model uses pretrained ResNet18 encoder weights and fine-tunes on the
FiveK subset.

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pt --split test
```

Evaluation reports PSNR and SSIM. Visual comparison grids are saved to
`outputs/comparisons/` with:

```text
Original | Model output | Expert D target
```

## Use on New Photos

Enhance one new image:

```bash
python predict.py --input path/to/photo.jpg --checkpoint checkpoints/best.pt
```

Enhance every image in a folder:

```bash
python predict.py --input path/to/photos --checkpoint checkpoints/best.pt
```

Enhanced images are saved to `outputs/predictions/`.

## Research Question

Can a neural network learn to automatically enhance original photos to match
Expert D's professional retouching style?
