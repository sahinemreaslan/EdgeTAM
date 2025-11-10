# EdgeTAM Deployment Guide

This directory contains tools and scripts for deploying EdgeTAM models in production environments.

## Overview

EdgeTAM can be deployed in multiple formats:
- **ONNX**: Cross-platform, good performance, easy to use
- **TensorRT**: NVIDIA GPUs only, best performance, requires CUDA

## Quick Start

### 1. Export to ONNX

First, export the PyTorch model to ONNX format:

```bash
python export_to_onnx.py --checkpoint checkpoints/edgetam.pt --output-dir onnx_models
```

**Options:**
- `--checkpoint`: Path to EdgeTAM checkpoint (default: `checkpoints/edgetam.pt`)
- `--config`: Path to config file (default: `configs/edgetam.yaml`)
- `--output-dir`: Output directory (default: `onnx_models`)
- `--opset-version`: ONNX opset version (default: 18)
- `--verify`: Verify exported models

**Output:**
- `onnx_models/edgetam_image_encoder.onnx`: Image encoder
- `onnx_models/edgetam_mask_decoder.onnx`: Mask decoder

### 2. Verify ONNX Models

```bash
python export_to_onnx.py --checkpoint checkpoints/edgetam.pt --verify
```

### 3. Run ONNX Inference

```bash
# Point-based segmentation
python deploy/simple_inference.py \
    --onnx-dir onnx_models \
    --image examples/truck.jpg \
    --point 512,512 \
    --output output.png

# Box-based segmentation
python deploy/simple_inference.py \
    --onnx-dir onnx_models \
    --image examples/truck.jpg \
    --box 100,100,900,900 \
    --output output.png

# Multiple points
python deploy/simple_inference.py \
    --onnx-dir onnx_models \
    --image examples/truck.jpg \
    --point 400,400 \
    --point 600,600 \
    --output output.png
```

## TensorRT Deployment (NVIDIA GPUs)

### Prerequisites

```bash
# Install TensorRT and PyCUDA
pip install tensorrt pycuda

# Or use NVIDIA's container
docker pull nvcr.io/nvidia/tensorrt:23.12-py3
```

### 1. Convert to TensorRT

```bash
# FP32 (full precision)
python convert_to_tensorrt.py --onnx-dir onnx_models --output-dir tensorrt_models

# FP16 (faster, slightly less accurate)
python convert_to_tensorrt.py --onnx-dir onnx_models --output-dir tensorrt_models --fp16

# Custom workspace size
python convert_to_tensorrt.py --onnx-dir onnx_models --workspace 8  # 8GB
```

**Options:**
- `--onnx-dir`: Directory with ONNX models
- `--output-dir`: Output directory (default: `tensorrt_models`)
- `--fp16`: Enable FP16 mode
- `--int8`: Enable INT8 mode (requires calibration)
- `--workspace`: Workspace size in GB (default: 4)
- `--verify`: Verify engines after conversion

### 2. Run TensorRT Inference

```bash
python deploy/simple_inference.py \
    --engine-dir tensorrt_models \
    --image examples/truck.jpg \
    --point 512,512 \
    --output output.png
```

## Benchmarking

### Benchmark ONNX Runtime

```bash
python deploy/benchmark.py --onnx-dir onnx_models --iterations 100
```

### Benchmark TensorRT

```bash
python deploy/benchmark.py --engine-dir tensorrt_models --iterations 100
```

### Compare Backends

```bash
python deploy/benchmark.py \
    --onnx-dir onnx_models \
    --engine-dir tensorrt_models \
    --compare
```

## Model Architecture

EdgeTAM is exported in two parts:

### 1. Image Encoder

**Inputs:**
- `image`: `[B, 3, 1024, 1024]` - Input image (normalized to [0, 1])

**Outputs:**
- `image_embeddings`: `[B, 256, 64, 64]` - Image features
- `high_res_feat_0`: `[B, 32, 256, 256]` - High-res features (level 0)
- `high_res_feat_1`: `[B, 64, 128, 128]` - High-res features (level 1)

### 2. Mask Decoder

**Inputs:**
- `image_embeddings`: `[B, 256, 64, 64]` - From image encoder
- `point_coords`: `[B, N, 2]` - Point coordinates (x, y) in [0, 1024]
- `point_labels`: `[B, N]` - Point labels (1=foreground, 0=background, 2/3=box)
- `high_res_feat_0`: `[B, 32, 256, 256]` - High-res features (optional)
- `high_res_feat_1`: `[B, 64, 128, 128]` - High-res features (optional)

**Outputs:**
- `masks`: `[B, 1, 1024, 1024]` - Predicted masks (probabilities)
- `iou_predictions`: `[B, 1]` - IoU confidence scores

## Prompting

EdgeTAM supports multiple prompting modes:

### Point Prompts

```python
points = [[x1, y1], [x2, y2], ...]
labels = [1, 1, ...]  # 1 = foreground, 0 = background
```

### Box Prompts

```python
# Box as two corner points
points = [[x1, y1], [x2, y2]]
labels = [2, 3]  # 2 = top-left, 3 = bottom-right
```

### Combined Prompts

```python
# Box + additional point
points = [[x1, y1], [x2, y2], [x3, y3]]
labels = [2, 3, 1]  # Box corners + foreground point
```

## Integration Examples

### Python Integration

```python
from deploy.simple_inference import ONNXPredictor
from PIL import Image

# Initialize predictor
predictor = ONNXPredictor('onnx_models')

# Load image
image = Image.open('image.jpg')

# Predict mask
points = [[512, 512]]
labels = [1]
mask, iou = predictor.predict(image, points, labels)

# mask is a [1024, 1024] numpy array
print(f"IoU confidence: {iou:.4f}")
```

### ONNX Runtime Direct Usage

```python
import onnxruntime as ort
import numpy as np

# Load models
encoder = ort.InferenceSession('onnx_models/edgetam_image_encoder.onnx')
decoder = ort.InferenceSession('onnx_models/edgetam_mask_decoder.onnx')

# Prepare image (normalized to [0, 1])
image = np.random.rand(1, 3, 1024, 1024).astype(np.float32)

# Encode
embeddings, high_res_0, high_res_1 = encoder.run(None, {'image': image})

# Prepare prompts
point_coords = np.array([[[512, 512]]], dtype=np.float32)
point_labels = np.array([[1]], dtype=np.int32)

# Decode
masks, iou = decoder.run(None, {
    'image_embeddings': embeddings,
    'point_coords': point_coords,
    'point_labels': point_labels,
    'high_res_feat_0': high_res_0,
    'high_res_feat_1': high_res_1,
})
```

## Performance Tips

### 1. Batch Processing

The models support dynamic batch sizes. Process multiple images in a batch for better throughput:

```python
# Encode multiple images at once
images = np.stack([image1, image2, image3])  # [3, 3, 1024, 1024]
embeddings, high_res_0, high_res_1 = encoder.run(None, {'image': images})
```

### 2. Reuse Embeddings

If you need multiple masks for the same image, encode once and reuse:

```python
# Encode once
embeddings, high_res_0, high_res_1 = predictor.encode_image(image)

# Decode multiple times with different prompts
for points, labels in prompt_list:
    mask, iou = predictor.predict_mask(embeddings, points, labels, high_res_0, high_res_1)
```

### 3. Execution Providers

For ONNX Runtime, choose the best execution provider:

```python
# CUDA (NVIDIA GPU)
session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

# TensorRT (NVIDIA GPU, fastest)
session = ort.InferenceSession(model_path, providers=['TensorrtExecutionProvider'])

# CPU
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
```

### 4. Disable High-Res Features

If speed is more important than accuracy, disable high-res features:

```bash
python deploy/simple_inference.py --onnx-dir onnx_models --no-high-res --image image.jpg --point 512,512
```

## Troubleshooting

### ONNX Export Issues

**Issue:** ONNX export fails with opset version error
```bash
# Try a different opset version
python export_to_onnx.py --opset-version 17
```

**Issue:** Model verification fails
```bash
# Install verification tools
pip install onnx onnxruntime
```

### TensorRT Issues

**Issue:** TensorRT not found
```bash
# Check CUDA installation
nvidia-smi

# Install TensorRT
pip install tensorrt

# Or use NVIDIA container
docker pull nvcr.io/nvidia/tensorrt:23.12-py3
```

**Issue:** Out of memory during conversion
```bash
# Reduce workspace size
python convert_to_tensorrt.py --onnx-dir onnx_models --workspace 2
```

### Inference Issues

**Issue:** Slow inference on CPU
- Use GPU with CUDA execution provider
- Enable TensorRT execution provider
- Reduce image resolution

**Issue:** Low-quality masks
- Ensure image is properly normalized to [0, 1]
- Check point coordinates are in correct format
- Enable high-res features (default)

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- CPU with AVX2 support

### Recommended for Production
- NVIDIA GPU (RTX 2060 or better)
- 8GB+ VRAM
- CUDA 11.8+
- TensorRT 8.6+

## Performance Expectations

On NVIDIA RTX 4090:

| Backend | Precision | Latency | Throughput |
|---------|-----------|---------|------------|
| ONNX    | FP32      | ~50ms   | ~20 FPS    |
| TensorRT| FP32      | ~15ms   | ~65 FPS    |
| TensorRT| FP16      | ~8ms    | ~120 FPS   |

*Note: Performance varies by hardware and image complexity*

## License

See the main repository LICENSE file for licensing information.

## Support

For issues and questions:
- Check the main [README](../README.md)
- Open an issue on GitHub
- See [examples](../examples/) for more use cases
