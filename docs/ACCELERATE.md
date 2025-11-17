# Hugging Face Accelerate Integration

Depth Anything 3 now supports [Hugging Face Accelerate](https://huggingface.co/docs/accelerate) for optimized inference. Accelerate provides several benefits for deep learning inference:

## Benefits

- **Automatic Mixed Precision**: Automatically uses FP16 or BF16 precision based on hardware capabilities
- **Better Memory Management**: Optimized memory usage for large models
- **Multi-GPU Support**: Easy distributed inference across multiple GPUs
- **Device Placement**: Intelligent device placement and data handling
- **Zero Configuration**: Works out of the box with sensible defaults

## Installation

Accelerate is included as a dependency when you install Depth Anything 3:

```bash
pip install -e .
```

Or install it separately:

```bash
pip install accelerate>=0.20.0
```

## Usage

### Basic Usage

Simply add `use_accelerate=True` when creating the model:

```python
import torch
from depth_anything_3.api import DepthAnything3

# Standard initialization
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")

# With Accelerate optimization
model = DepthAnything3.from_pretrained(
    "depth-anything/DA3-LARGE",
    use_accelerate=True
)

# The model can be used the same way as before
prediction = model.inference(images, export_dir="output", export_format="glb")
```

### Custom Accelerator Configuration

You can pass additional arguments to customize the Accelerator:

```python
model = DepthAnything3.from_pretrained(
    "depth-anything/DA3-LARGE",
    use_accelerate=True,
    mixed_precision="bf16",  # Force BF16 precision
    device_placement=True,
    # Any other Accelerator arguments...
)
```

### Multi-GPU Inference

Accelerate makes multi-GPU inference simple:

```python
# The Accelerator will automatically detect and use available GPUs
model = DepthAnything3.from_pretrained(
    "depth-anything/DA3-GIANT",
    use_accelerate=True
)

# Your inference code remains the same
prediction = model.inference(images)
```

## Performance Considerations

### When to Use Accelerate

Accelerate is most beneficial when:
- Working with large models (DA3-GIANT, DA3-LARGE)
- Processing large batches of images
- Using multiple GPUs
- Memory is a constraint
- You want automatic mixed precision without manual configuration

### Backward Compatibility

The `use_accelerate` parameter is **optional** and defaults to `False`. This ensures:
- Existing code continues to work without changes
- Users can opt-in to Accelerate optimization
- Fallback to standard inference if Accelerate is not available

## Examples

### Example 1: Monocular Depth Estimation

```python
import glob
import os
from depth_anything_3.api import DepthAnything3

# Load model with Accelerate
model = DepthAnything3.from_pretrained(
    "depth-anything/DA3-LARGE",
    use_accelerate=True
)

# Process images
images = sorted(glob.glob("path/to/images/*.png"))
prediction = model.inference(
    images,
    export_dir="output",
    export_format="glb"
)

print(f"Depth shape: {prediction.depth.shape}")
print(f"Extrinsics shape: {prediction.extrinsics.shape}")
```

### Example 2: Multi-View Depth with Poses

```python
import numpy as np
from depth_anything_3.api import DepthAnything3

# Load model with Accelerate
model = DepthAnything3.from_pretrained(
    "depth-anything/DA3NESTED-GIANT-LARGE",
    use_accelerate=True
)

# Your images and poses
images = ["img1.png", "img2.png", "img3.png"]
extrinsics = np.load("extrinsics.npy")  # (N, 4, 4)
intrinsics = np.load("intrinsics.npy")  # (N, 3, 3)

# Run inference
prediction = model.inference(
    images,
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    export_dir="output",
    export_format="glb-npz"
)
```

### Example 3: Video Processing

```python
from depth_anything_3.api import DepthAnything3

# Load model with Accelerate for faster video processing
model = DepthAnything3.from_pretrained(
    "depth-anything/DA3-LARGE",
    use_accelerate=True
)

# Process video frames
video_frames = ["frame_0001.png", "frame_0002.png", ...]
prediction = model.inference(
    video_frames,
    process_res=504,
    export_dir="video_output",
    export_format="glb"
)
```

## Troubleshooting

### Accelerate Not Available

If you see a warning that Accelerate is not available:

```bash
pip install accelerate>=0.20.0
```

### Memory Issues

If you encounter out-of-memory errors even with Accelerate, try:

1. Reduce batch size (process fewer images at once)
2. Reduce `process_res` parameter
3. Use a smaller model (e.g., DA3-BASE instead of DA3-GIANT)

### Mixed Precision Issues

If you encounter numerical issues with mixed precision, you can disable it:

```python
model = DepthAnything3.from_pretrained(
    "depth-anything/DA3-LARGE",
    use_accelerate=True,
    mixed_precision="no"  # Disable mixed precision
)
```

## Technical Details

When `use_accelerate=True`:

1. An `Accelerator` instance is created with sensible defaults:
   - `mixed_precision`: "bf16" if supported by hardware, otherwise "fp16"
   - `device_placement`: True (automatic device management)

2. The model is prepared using `accelerator.prepare()`:
   - Wraps the model for optimized execution
   - Handles device placement automatically
   - Manages mixed precision contexts

3. Forward passes use Accelerate's automatic mixed precision:
   - No manual `torch.autocast` needed
   - Automatic dtype conversions
   - Optimized memory usage

## Comparison: Standard vs Accelerate

| Feature | Standard | With Accelerate |
|---------|----------|-----------------|
| Mixed Precision | Manual `torch.autocast` | Automatic |
| Multi-GPU | Manual setup required | Automatic |
| Device Placement | Manual `.to(device)` | Automatic |
| Memory Management | Standard PyTorch | Optimized |
| Configuration | More code | Less code |

## References

- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [Accelerate Quick Tour](https://huggingface.co/docs/accelerate/quicktour)
- [Mixed Precision Training](https://huggingface.co/docs/accelerate/usage_guides/mixed_precision)
