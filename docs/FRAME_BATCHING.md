# Frame Batching Feature

## Overview

This feature adds support for processing frames in batches to prevent Out-of-Memory (OOM) errors when working with many frames. When processing a large number of frames (e.g., video with hundreds of frames), the model previously attempted to extract features for all frames simultaneously, which could exceed available GPU memory.

## Usage

### API Parameter

The `inference()` method now accepts an optional `frame_batch_size` parameter:

```python
from depth_anything_3.api import DepthAnything3
import torch

# Load model
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)

# Process images with frame batching
images = [...]  # List of many images (e.g., 100+ frames)

# Process all frames at once (default, best quality but may OOM)
prediction = model.inference(images)

# Process frames in batches of 10 (reduces memory usage)
prediction = model.inference(images, frame_batch_size=10)

# Process frames in batches of 5 (further reduces memory usage)
prediction = model.inference(images, frame_batch_size=5)
```

### When to Use Frame Batching

**Use `frame_batch_size` when:**
- Processing many frames (50+) and encountering OOM errors
- Working with high-resolution images
- Using limited GPU memory
- Processing video sequences with hundreds or thousands of frames

**Don't use `frame_batch_size` (or set it to None) when:**
- Processing few frames (< 20)
- Maximum quality is required and memory is sufficient
- GPU memory is abundant

### Recommended Settings

| Number of Frames | GPU Memory | Recommended frame_batch_size |
|------------------|------------|------------------------------|
| < 20             | Any        | None (default)               |
| 20-50            | 8GB        | None or 20                   |
| 50-100           | 8GB        | 10-20                        |
| 100-200          | 8GB        | 5-10                         |
| 200+             | 8GB        | 5-10                         |
| Any              | 16GB+      | None or higher values        |

## How It Works

### Without Frame Batching (Default)
```
All frames → Feature Extraction → Output
[1,2,3,...,N]    (processes all N frames together)
```

### With Frame Batching
```
Batch 1 [1-5]   → Feature Extraction → Output 1
Batch 2 [6-10]  → Feature Extraction → Output 2
Batch 3 [11-15] → Feature Extraction → Output 3
...
→ Concatenate all outputs → Final Output
```

### Technical Details

1. **Frame Splitting**: Input frames are split into batches of size `frame_batch_size`
2. **Independent Processing**: Each batch is processed independently through all model layers
3. **Output Concatenation**: Results from all batches are concatenated to produce the final output

### Trade-offs

**Advantages:**
- ✓ Prevents OOM errors with many frames
- ✓ Enables processing of arbitrarily large frame sequences
- ✓ Allows working with limited GPU memory
- ✓ No quality loss for local processing (patch embedding, local attention)

**Limitations:**
- ✗ May reduce cross-batch frame consistency in global attention layers
- ✗ Each batch processes global attention independently
- ✗ Slightly slower due to batching overhead

**Quality Impact:**
- For most use cases, the quality impact is minimal as local attention (per-frame processing) remains unchanged
- Global attention layers process each batch independently, which may slightly reduce multi-view consistency across batch boundaries
- For best quality, set `frame_batch_size=None` or equal to the total number of frames

## Examples

### Example 1: Processing a Long Video Sequence

```python
import glob
import torch
from depth_anything_3.api import DepthAnything3

# Load model
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
model = model.to(device=device)

# Load video frames (e.g., 300 frames)
frames = sorted(glob.glob("video_frames/*.png"))
print(f"Processing {len(frames)} frames")

# Process with frame batching to avoid OOM
prediction = model.inference(
    frames,
    frame_batch_size=10,  # Process 10 frames at a time
    export_dir="output",
    export_format="npz"
)

print(f"Depth shape: {prediction.depth.shape}")  # (300, H, W)
```

### Example 2: Adjusting Batch Size Based on Available Memory

```python
import torch
from depth_anything_3.api import DepthAnything3

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
model = model.to(device=device)

images = [...]  # Your image list

# Get available GPU memory
if torch.cuda.is_available():
    free_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Available GPU memory: {free_memory_gb:.1f} GB")

    # Adjust batch size based on available memory
    if free_memory_gb < 8:
        frame_batch_size = 5
    elif free_memory_gb < 16:
        frame_batch_size = 10
    else:
        frame_batch_size = None  # Process all at once
else:
    frame_batch_size = 5  # Conservative for CPU

print(f"Using frame_batch_size: {frame_batch_size}")
prediction = model.inference(images, frame_batch_size=frame_batch_size)
```

### Example 3: Comparing Quality

```python
from depth_anything_3.api import DepthAnything3
import torch
import numpy as np

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
model = model.to(device=device)

images = [...]  # Your images (e.g., 30 frames)

# Process without batching (if memory permits)
pred_no_batch = model.inference(images, frame_batch_size=None)

# Process with batching
pred_batched = model.inference(images, frame_batch_size=10)

# Compare outputs
depth_diff = np.abs(pred_no_batch.depth - pred_batched.depth)
mean_diff = np.mean(depth_diff)
max_diff = np.max(depth_diff)

print(f"Mean depth difference: {mean_diff}")
print(f"Max depth difference: {max_diff}")
```

## Implementation Details

The batching is implemented at the vision transformer level in the `_get_intermediate_layers_not_chunked` method. The key aspects are:

1. **Patch Embedding**: Each batch is independently embedded into patches
2. **Attention Layers**: Both local and global attention layers process each batch independently
3. **Feature Concatenation**: Outputs from all batches are concatenated along the frame dimension
4. **Auxiliary Features**: If requested, auxiliary features from intermediate layers are also properly batched and concatenated

## Troubleshooting

### Still Getting OOM Errors
- Reduce `frame_batch_size` further (try 3-5)
- Reduce `process_res` parameter
- Use a smaller model variant (e.g., DA3-SMALL instead of DA3-GIANT)

### Quality Issues
- Increase `frame_batch_size` to include more frames per batch
- Set `frame_batch_size=None` if memory permits
- Ensure sufficient overlap between sequential frames

### Performance Issues
- Batching adds some overhead; if memory is not an issue, use `frame_batch_size=None`
- Larger batch sizes are more efficient (less overhead)
- Consider processing on GPU vs CPU based on available resources
