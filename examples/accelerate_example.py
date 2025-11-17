#!/usr/bin/env python3
"""
Example script demonstrating Hugging Face Accelerate integration with Depth Anything 3.

This script compares standard inference with Accelerate-optimized inference.
"""

import time
import torch
import numpy as np
from PIL import Image
from depth_anything_3.api import DepthAnything3


def create_dummy_image(height=504, width=672):
    """Create a dummy RGB image for testing."""
    return Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))


def benchmark_inference(use_accelerate=False, num_images=3, num_runs=3):
    """Benchmark inference with and without Accelerate."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking {'WITH' if use_accelerate else 'WITHOUT'} Accelerate")
    print(f"{'=' * 60}")

    # Create model
    print("Loading model...")
    model = DepthAnything3(
        model_name="da3-small",  # Use smallest model for quick testing
        use_accelerate=use_accelerate
    )

    # Check if CUDA is available
    if torch.cuda.is_available():
        if not use_accelerate or not model.accelerator:
            model = model.to("cuda")
        print(f"Device: {model._get_model_device()}")
    else:
        print("CUDA not available, using CPU")

    # Create dummy images
    images = [create_dummy_image() for _ in range(num_images)]

    # Warmup run
    print("Performing warmup run...")
    _ = model.inference(images, process_res=252)

    # Benchmark runs
    times = []
    print(f"\nRunning {num_runs} inference iterations...")
    for i in range(num_runs):
        start_time = time.time()
        prediction = model.inference(images, process_res=252)
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")

    # Results
    avg_time = np.mean(times)
    std_time = np.std(times)
    print("\nResults:")
    print(f"  Average time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"  Depth shape: {prediction.depth.shape}")
    print(f"  Extrinsics shape: {prediction.extrinsics.shape}")

    return avg_time, times


def main():
    """Main function to run benchmarks."""
    print("Depth Anything 3 - Accelerate Integration Example")
    print("=" * 60)

    # Check Accelerate availability
    try:
        import accelerate  # noqa: F401
        print("✓ Accelerate is available")
    except ImportError:
        print("✗ Accelerate is NOT available")
        print("  Install with: pip install accelerate>=0.20.0")
        return

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
        print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        print("✗ CUDA is not available, using CPU")

    # Run benchmarks
    num_images = 3
    num_runs = 3

    # Standard inference
    time_standard, times_standard = benchmark_inference(
        use_accelerate=False,
        num_images=num_images,
        num_runs=num_runs
    )

    # Accelerate inference
    time_accelerate, times_accelerate = benchmark_inference(
        use_accelerate=True,
        num_images=num_images,
        num_runs=num_runs
    )

    # Comparison
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"Standard:   {time_standard:.3f}s")
    print(f"Accelerate: {time_accelerate:.3f}s")
    speedup = time_standard / time_accelerate
    print(f"Speedup:    {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")

    print("\nNote: Speedup varies based on:")
    print("  - Hardware (GPU model, CUDA version)")
    print("  - Model size (larger models benefit more)")
    print("  - Batch size (larger batches benefit more)")
    print("  - Mixed precision support (BF16 > FP16 > FP32)")


if __name__ == "__main__":
    main()
