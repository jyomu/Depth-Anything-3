# Depth Anything 3 Examples

This directory contains example scripts demonstrating various usage patterns for Depth Anything 3.

## VRAM Efficient Reconstruction

`vram_efficient_reconstruction.py` demonstrates how to process large scenes with limited VRAM (under 12GB).

### Features
- Automatic chunking of large scenes
- VRAM monitoring and reporting
- Graceful OOM handling with suggestions
- Intermediate result saving
- Merged scene reconstruction

### Usage

Basic usage:
```bash
python examples/vram_efficient_reconstruction.py assets/examples/SOH --output ./output
```

Custom chunk size and resolution:
```bash
python examples/vram_efficient_reconstruction.py /path/to/images \
    --output ./output \
    --chunk-size 4 \
    --process-res 448
```

Using a smaller model:
```bash
python examples/vram_efficient_reconstruction.py /path/to/images \
    --output ./output \
    --model depth-anything/DA3-BASE
```

### Arguments

- `input`: Input image directory or glob pattern
- `--output, -o`: Output directory (required)
- `--model`: Model name (default: depth-anything/DA3-LARGE)
- `--chunk-size`: Number of frames per chunk (default: 8)
- `--process-res`: Processing resolution (default: 504)
- `--max-vram`: Target VRAM limit in GB (default: 11.0)
- `--export-format`: Export format (default: mini_npz)
- `--num-max-points`: Maximum number of points (default: 500,000)
- `--conf-thresh-percentile`: Confidence threshold percentile (default: 50.0)
- `--device`: Device to use (default: cuda)
- `--quiet`: Suppress verbose logging

### Output

The script creates:
- `chunk_XXXX/` directories for each chunk
- `merged_scene.npz` containing the combined results
- VRAM usage reports for monitoring

## See Also

- [VRAM Optimization Report (Japanese)](../docs/VRAM_OPTIMIZATION_REPORT_JP.md) - Comprehensive guide on VRAM optimization strategies
- [API Documentation](../docs/API.md) - Python API reference
- [CLI Documentation](../docs/CLI.md) - Command-line interface guide
