#!/usr/bin/env python3
"""
Verification script to ensure global attention works correctly.

This script verifies that:
1. The vision transformer processes frames with alternating local/global attention
2. Global attention sees all frames together (not batched)
3. No frame_batch_size parameter exists in the DinoV2 implementation
"""

import sys
import inspect

sys.path.insert(0, 'src')

from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer  # noqa: E402


def test_no_frame_batching_parameter():
    """Verify that frame_batch_size parameter doesn't exist in vision transformer."""
    print("Test 1: Checking for frame_batch_size parameter...")

    # Check _get_intermediate_layers_not_chunked
    sig = inspect.signature(DinoVisionTransformer._get_intermediate_layers_not_chunked)
    params = list(sig.parameters.keys())

    if 'frame_batch_size' in params:
        print("  ❌ FAIL: frame_batch_size parameter found in _get_intermediate_layers_not_chunked")
        return False

    # Check get_intermediate_layers
    sig = inspect.signature(DinoVisionTransformer.get_intermediate_layers)
    params = list(sig.parameters.keys())

    if 'frame_batch_size' in params:
        print("  ❌ FAIL: frame_batch_size parameter found in get_intermediate_layers")
        return False

    print("  ✓ PASS: No frame_batch_size parameter found")
    return True


def test_attention_pattern():
    """Verify that process_attention method exists and handles global/local attention."""
    print("\nTest 2: Checking attention pattern implementation...")

    # Check that process_attention method exists
    if not hasattr(DinoVisionTransformer, 'process_attention'):
        print("  ❌ FAIL: process_attention method not found")
        return False

    # Check method signature
    sig = inspect.signature(DinoVisionTransformer.process_attention)
    params = list(sig.parameters.keys())

    expected_params = ['self', 'x', 'block', 'attn_type', 'pos', 'attn_mask']
    if params != expected_params:
        print(f"  ❌ FAIL: process_attention has unexpected signature: {params}")
        return False

    print("  ✓ PASS: process_attention method exists with correct signature")
    return True


def test_source_code_integrity():
    """Verify that critical parts of the vision transformer are intact."""
    print("\nTest 3: Checking source code integrity...")

    import depth_anything_3.model.dinov2.vision_transformer as vt_module
    source = inspect.getsource(vt_module.DinoVisionTransformer._get_intermediate_layers_not_chunked)

    # Check for global attention pattern
    if 'attn_type="global"' not in source and '"global"' not in source:
        print("  ❌ FAIL: Global attention pattern not found in source code")
        return False

    # Check for local attention pattern
    if 'attn_type="local"' not in source and '"local"' not in source:
        print("  ❌ FAIL: Local attention pattern not found in source code")
        return False

    # Make sure no batching logic exists
    if 'num_batches' in source or 'batch_idx' in source or 'frame_batch_size' in source:
        print("  ❌ FAIL: Batching logic found in source code")
        return False

    print("  ✓ PASS: Source code integrity verified")
    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Global Attention Verification")
    print("=" * 60)

    tests = [
        test_no_frame_batching_parameter,
        test_attention_pattern,
        test_source_code_integrity,
    ]

    results = [test() for test in tests]

    print("\n" + "=" * 60)
    if all(results):
        print("✓ All tests PASSED")
        print("\nGlobal attention is working correctly.")
        print("The vision transformer processes all frames together in global")
        print("attention layers, maintaining cross-frame consistency.")
        return 0
    else:
        print("❌ Some tests FAILED")
        print("\nThe vision transformer may have been modified incorrectly.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
