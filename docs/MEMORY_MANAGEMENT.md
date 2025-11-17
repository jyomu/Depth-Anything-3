# Memory Management and Global Attention

## Overview

This document explains memory management considerations in Depth Anything 3, particularly regarding processing many frames and why frame batching cannot be implemented without compromising model quality.

## The Problem: OOM with Many Frames

When processing many frames (e.g., 100+ frames from a video), the model can run out of GPU memory. This happens because:

1. **Patch Embedding**: Each frame is split into patches, creating many tokens: `(B*S) × (H/14) × (W/14) × embed_dim`
2. **Attention Computation**: Attention layers compute pairwise similarities between tokens
3. **Memory Growth**: Memory usage scales with the number of frames

## Why Frame Batching Cannot Be Used

### PR #5 Attempted Solution

PR #5 (https://github.com/jyomu/Depth-Anything-3/pull/5) attempted to add frame batching by splitting frames into smaller batches and processing them independently through the vision transformer. However, **this approach fundamentally breaks the model's architecture**.

### The Critical Issue: Global Attention Loss

Depth Anything 3's vision transformer uses an **alternating attention pattern**:

- **Local Attention** (even layers): Processes each frame independently
  - Reshapes to `(b*s) n c` where each frame is processed separately
  - ✓ Can be batched without quality loss
  
- **Global Attention** (odd layers after `alt_start`): Processes all frames together
  - Reshapes to `b (s*n) c` where all frames see each other
  - ✗ **CANNOT be batched** - frames must see each other for cross-frame consistency

### Why Global Attention Matters

Global attention is crucial for:

1. **Multi-view Consistency**: Ensures depth predictions are consistent across different views
2. **Cross-frame Correspondence**: Enables the model to understand relationships between frames
3. **Pose Estimation**: Camera pose estimation requires understanding relationships between multiple views
4. **3D Reconstruction**: Building coherent 3D geometry requires cross-frame information

**When frame batching is applied**, global attention only sees frames within each batch, not across all frames. This breaks the fundamental architecture of the model and degrades quality significantly.

### Code Location

The attention mechanism is implemented in:
- `src/depth_anything_3/model/dinov2/vision_transformer.py`
  - Method: `process_attention()`
  - Lines showing alternating pattern: Lines 317-323

```python
# Global attention: all frames see each other
if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
    x = self.process_attention(
        x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
    )
# Local attention: per-frame processing
else:
    x = self.process_attention(x, blk, "local", pos=l_pos)
```

## Design Principle: Don't Modify Feature Extraction

The DinoV2 vision transformer is used for feature extraction and is not the primary subject of this repository. Modifying its behavior (e.g., adding frame batching) can:

1. Break the pretrained weights and attention patterns
2. Compromise the model's ability to extract consistent features
3. Degrade performance on all downstream tasks

**Therefore, the DinoV2 implementation must remain unchanged.**

## Recommended Solutions for OOM Issues

If you encounter out-of-memory errors when processing many frames, consider these alternatives:

### 1. Reduce Number of Frames
Process fewer frames at a time:
```python
# Instead of processing all 200 frames at once
frames = load_frames(...)  # 200 frames

# Process in sequential batches and save results
for i in range(0, len(frames), 20):
    batch = frames[i:i+20]
    prediction = model.inference(batch)
    save_prediction(prediction, f"batch_{i}")
```

**Note**: This approach processes each batch independently, so there will be no cross-batch consistency. However, within each batch, global attention works correctly.

### 2. Use Smaller Model Variants
Smaller models require less memory:
- `DA3-GIANT` (1.15B params) → `DA3-LARGE` (0.35B params)
- `DA3-LARGE` → `DA3-BASE` (0.12B params)
- `DA3-BASE` → `DA3-SMALL` (0.08B params)

### 3. Reduce Processing Resolution
Lower resolution requires less memory:
```python
# Default: process_res=504
prediction = model.inference(images, process_res=448)  # Reduces memory usage

# Or use a different resize method
prediction = model.inference(
    images, 
    process_res=504,
    process_res_method="lower_bound_resize"  # More aggressive downsampling
)
```

### 4. Upgrade Hardware
- Use GPUs with more VRAM (e.g., A100 with 40GB/80GB)
- Use multiple GPUs (though this requires model parallelism implementation)

### 5. Process Frames Sequentially
For applications that don't require multi-view consistency:
```python
# Process one frame at a time (no global attention across frames anyway)
results = []
for frame in frames:
    prediction = model.inference([frame])  # Single frame
    results.append(prediction)
```

This is suitable for:
- Monocular depth estimation on independent images
- Video processing where temporal consistency is not critical

## Alternative Architectures (Future Work)

To truly solve the OOM issue while maintaining quality, alternative approaches would require:

1. **Memory-Efficient Attention**: Implement Flash Attention or other efficient attention mechanisms
   - Reduces memory complexity of attention from O(n²) to O(n log n) or O(n)
   - Requires significant implementation work

2. **Hierarchical Processing**: Process frames in a two-stage approach
   - Stage 1: Extract local features per-frame (low memory)
   - Stage 2: Global attention on compressed representations (high quality)
   - Requires architecture redesign

3. **Streaming/Progressive Processing**: Design for incremental frame processing
   - Maintain a sliding window of frames
   - Update global understanding progressively
   - Requires new training paradigm

These approaches would require substantial research and development effort and are beyond the scope of simple memory optimization.

## Conclusion

**Frame batching cannot be implemented in Depth Anything 3 without breaking global attention**, which is essential for the model's multi-view capabilities. PR #5's approach, while well-intentioned, fundamentally compromises the model architecture.

Users experiencing OOM issues should:
1. Use the recommended solutions above (reduce frames, use smaller models, lower resolution)
2. Accept the memory-quality trade-off inherent in the current architecture
3. Wait for future memory-efficient implementations if maintaining quality is critical

The integrity of the vision transformer's attention mechanism must be preserved to ensure the model works as designed.
