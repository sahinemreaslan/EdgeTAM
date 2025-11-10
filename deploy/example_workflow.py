#!/usr/bin/env python3
"""
EdgeTAM Complete Workflow Example

This script demonstrates the complete workflow from model export to inference.

Usage:
    python deploy/example_workflow.py
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and print status"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n✗ Command failed with exit code {result.returncode}")
        return False

    print(f"\n✓ {description} completed successfully")
    return True


def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              EdgeTAM Deployment Workflow Example              ║
╚═══════════════════════════════════════════════════════════════╝

This script will guide you through the complete EdgeTAM deployment workflow:
1. Export PyTorch model to ONNX
2. Verify ONNX models
3. Run inference with ONNX
4. (Optional) Convert to TensorRT
5. Benchmark performance

""")

    # Check if checkpoint exists
    checkpoint_path = Path("checkpoints/edgetam.pt")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please download the checkpoint first:")
        print("  cd checkpoints && bash download_ckpts.sh")
        sys.exit(1)

    # Check if we have an example image
    examples_dir = Path("examples")
    if not examples_dir.exists():
        print("Warning: examples/ directory not found")
        print("Creating a dummy image for testing...")
        examples_dir.mkdir(exist_ok=True)
        # Create a simple test image
        try:
            from PIL import Image
            import numpy as np
            img = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype('uint8'))
            img.save("examples/test.jpg")
            test_image = "examples/test.jpg"
        except ImportError:
            print("Error: PIL not installed. Please provide an example image.")
            sys.exit(1)
    else:
        # Find first image in examples
        image_files = list(examples_dir.glob("*.jpg")) + list(examples_dir.glob("*.png"))
        if image_files:
            test_image = str(image_files[0])
        else:
            print("Error: No test images found in examples/")
            sys.exit(1)

    print(f"Using test image: {test_image}")

    # Step 1: Export to ONNX
    if not run_command(
        "python export_to_onnx.py --checkpoint checkpoints/edgetam.pt --output-dir onnx_models",
        "Step 1: Export to ONNX"
    ):
        sys.exit(1)

    # Step 2: Verify ONNX models
    if not run_command(
        "python export_to_onnx.py --checkpoint checkpoints/edgetam.pt --verify",
        "Step 2: Verify ONNX Models"
    ):
        print("Warning: ONNX verification failed, but continuing...")

    # Step 3: Run ONNX inference
    if not run_command(
        f"python deploy/simple_inference.py --onnx-dir onnx_models --image {test_image} --point 256,256 --output output_onnx.png",
        "Step 3: Run ONNX Inference"
    ):
        sys.exit(1)

    # Step 4: Benchmark ONNX
    print("\nDo you want to run a performance benchmark? (y/n): ", end="")
    response = input().strip().lower()

    if response == 'y':
        if not run_command(
            "python deploy/benchmark.py --onnx-dir onnx_models --iterations 50",
            "Step 4: Benchmark ONNX Runtime"
        ):
            print("Warning: Benchmark failed, but continuing...")

    # Step 5: TensorRT conversion (optional)
    print("\nDo you want to convert to TensorRT? (requires NVIDIA GPU) (y/n): ", end="")
    response = input().strip().lower()

    if response == 'y':
        # Check if TensorRT is available
        try:
            import tensorrt
            has_tensorrt = True
        except ImportError:
            print("\nWarning: TensorRT not installed. Skipping TensorRT conversion.")
            print("To install TensorRT, visit: https://developer.nvidia.com/tensorrt")
            has_tensorrt = False

        if has_tensorrt:
            # Convert to TensorRT
            print("\nChoose precision mode:")
            print("1. FP32 (full precision)")
            print("2. FP16 (faster, slightly less accurate)")
            print("Choice (1 or 2): ", end="")
            precision = input().strip()

            fp16_flag = "--fp16" if precision == "2" else ""

            if run_command(
                f"python convert_to_tensorrt.py --onnx-dir onnx_models --output-dir tensorrt_models {fp16_flag}",
                "Step 5: Convert to TensorRT"
            ):
                # Run TensorRT inference
                if run_command(
                    f"python deploy/simple_inference.py --engine-dir tensorrt_models --image {test_image} --point 256,256 --output output_trt.png",
                    "Step 6: Run TensorRT Inference"
                ):
                    # Compare performance
                    run_command(
                        "python deploy/benchmark.py --onnx-dir onnx_models --engine-dir tensorrt_models --compare --iterations 50",
                        "Step 7: Compare ONNX vs TensorRT"
                    )

    # Summary
    print(f"\n{'='*60}")
    print("Workflow Complete!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print("  onnx_models/")
    print("    ├── edgetam_image_encoder.onnx")
    print("    └── edgetam_mask_decoder.onnx")

    if Path("tensorrt_models").exists():
        print("  tensorrt_models/")
        print("    ├── edgetam_image_encoder.trt")
        print("    └── edgetam_mask_decoder.trt")

    if Path("output_onnx.png").exists():
        print("  output_onnx.png (ONNX inference result)")

    if Path("output_trt.png").exists():
        print("  output_trt.png (TensorRT inference result)")

    print("\nNext steps:")
    print("  - Review deploy/README.md for more deployment options")
    print("  - Integrate the predictor into your application")
    print("  - Optimize for your specific use case")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
        sys.exit(1)
