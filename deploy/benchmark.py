#!/usr/bin/env python3
"""
EdgeTAM Benchmark Script

This script benchmarks the performance of EdgeTAM models in different formats.
Measures inference time, throughput, and memory usage.

Usage:
    python deploy/benchmark.py --onnx-dir onnx_models --iterations 100
    python deploy/benchmark.py --engine-dir tensorrt_models --iterations 100
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def benchmark_model(predictor, num_iterations=100, warmup_iterations=10):
    """
    Benchmark model performance

    Args:
        predictor: Model predictor instance
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        dict: Benchmark results
    """
    # Create dummy inputs
    dummy_image = Image.new('RGB', (1024, 1024), color='red')
    dummy_points = [[512, 512]]
    dummy_labels = [1]

    print(f"\nWarming up ({warmup_iterations} iterations)...")
    for i in range(warmup_iterations):
        _ = predictor.predict(dummy_image, dummy_points, dummy_labels)
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{warmup_iterations}")

    print(f"\nRunning benchmark ({num_iterations} iterations)...")
    times = []

    for i in range(num_iterations):
        start = time.time()
        mask, iou = predictor.predict(dummy_image, dummy_points, dummy_labels)
        end = time.time()

        times.append(end - start)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_iterations}")

    # Calculate statistics
    times = np.array(times)
    results = {
        'num_iterations': num_iterations,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'median_time': np.median(times),
        'p95_time': np.percentile(times, 95),
        'p99_time': np.percentile(times, 99),
        'throughput': 1.0 / np.mean(times),
    }

    return results


def print_benchmark_results(results, backend_name):
    """Print benchmark results in a formatted table"""
    print("\n" + "=" * 60)
    print(f"Benchmark Results: {backend_name}")
    print("=" * 60)
    print(f"Iterations: {results['num_iterations']}")
    print("-" * 60)
    print(f"Mean time:     {results['mean_time']*1000:8.2f} ms")
    print(f"Std dev:       {results['std_time']*1000:8.2f} ms")
    print(f"Min time:      {results['min_time']*1000:8.2f} ms")
    print(f"Max time:      {results['max_time']*1000:8.2f} ms")
    print(f"Median time:   {results['median_time']*1000:8.2f} ms")
    print(f"95th percentile: {results['p95_time']*1000:8.2f} ms")
    print(f"99th percentile: {results['p99_time']*1000:8.2f} ms")
    print("-" * 60)
    print(f"Throughput:    {results['throughput']:8.2f} FPS")
    print("=" * 60)


def compare_backends(onnx_results=None, trt_results=None):
    """Compare results from different backends"""
    if onnx_results is None or trt_results is None:
        return

    print("\n" + "=" * 60)
    print("Backend Comparison")
    print("=" * 60)

    onnx_time = onnx_results['mean_time'] * 1000
    trt_time = trt_results['mean_time'] * 1000
    speedup = onnx_time / trt_time

    print(f"ONNX Runtime:  {onnx_time:8.2f} ms")
    print(f"TensorRT:      {trt_time:8.2f} ms")
    print(f"Speedup:       {speedup:8.2f}x")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark EdgeTAM models")
    parser.add_argument(
        "--onnx-dir",
        type=str,
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--engine-dir",
        type=str,
        help="Directory containing TensorRT engines",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare ONNX and TensorRT backends (requires both --onnx-dir and --engine-dir)",
    )

    args = parser.parse_args()

    if not args.onnx_dir and not args.engine_dir:
        print("Error: Must specify either --onnx-dir or --engine-dir")
        sys.exit(1)

    # Import predictor classes
    sys.path.insert(0, str(Path(__file__).parent))
    from simple_inference import ONNXPredictor, TensorRTPredictor

    onnx_results = None
    trt_results = None

    # Benchmark ONNX
    if args.onnx_dir:
        print("=" * 60)
        print("Benchmarking ONNX Runtime")
        print("=" * 60)

        try:
            predictor = ONNXPredictor(args.onnx_dir)
            onnx_results = benchmark_model(
                predictor,
                num_iterations=args.iterations,
                warmup_iterations=args.warmup
            )
            print_benchmark_results(onnx_results, "ONNX Runtime")
        except Exception as e:
            print(f"✗ ONNX benchmark failed: {e}")
            if not args.compare:
                sys.exit(1)

    # Benchmark TensorRT
    if args.engine_dir:
        print("\n" + "=" * 60)
        print("Benchmarking TensorRT")
        print("=" * 60)

        try:
            predictor = TensorRTPredictor(args.engine_dir)
            trt_results = benchmark_model(
                predictor,
                num_iterations=args.iterations,
                warmup_iterations=args.warmup
            )
            print_benchmark_results(trt_results, "TensorRT")
        except Exception as e:
            print(f"✗ TensorRT benchmark failed: {e}")
            print("Note: TensorRT inference is partially implemented")
            if not args.compare:
                sys.exit(1)

    # Compare if both available
    if args.compare and onnx_results and trt_results:
        compare_backends(onnx_results, trt_results)

    print("\n✓ Benchmark completed!")


if __name__ == "__main__":
    main()
