#!/usr/bin/env python3
"""
Patch BEVFormer's test.py to support both GPU and CPU single-GPU evaluation.

The original BEVFormer test.py has `assert False` which blocks single-GPU
evaluation entirely. This script patches it to:
  - Use GPU if a compatible CUDA device (compute capability >= 5.0) is available
  - Fall back to CPU mode otherwise (slow but functional)

Usage:
    cd core_models/BEVFormer
    python ../../scripts/patch_bevformer_test.py

Or from project root:
    python scripts/patch_bevformer_test.py
"""

import re
import os
import sys


def find_test_py():
    """Locate BEVFormer's test.py relative to this script or cwd."""
    candidates = [
        "tools/test.py",  # if run from BEVFormer dir
        "core_models/BEVFormer/tools/test.py",  # if run from project root
    ]
    # Also try relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates.append(
        os.path.join(project_root, "core_models", "BEVFormer", "tools", "test.py")
    )

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def patch_test_py(filepath):
    """Apply the single-GPU evaluation patch to test.py."""
    with open(filepath, "r") as f:
        content = f.read()

    # Check if already patched
    if "torch.cuda.get_device_capability" in content:
        print(f"✅ {filepath} is already patched. No changes made.")
        return False

    # Check for the assert False pattern
    if "assert False" not in content:
        print(f"⚠️  No 'assert False' found in {filepath}. Skipping.")
        return False

    # Replace the broken single-GPU block
    old = re.compile(r"    if not distributed:.*?    else:", re.DOTALL)
    new_block = """    if not distributed:
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 5:
            model = MMDataParallel(model.cuda(), device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        else:
            # CPU fallback: manually unpack DataContainers
            from mmcv.parallel import DataContainer as DC
            model.eval()
            results = []
            prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
            for i, data in enumerate(data_loader):
                unpacked = {}
                for k, v in data.items():
                    if isinstance(v, DC):
                        unpacked[k] = v.data
                    elif isinstance(v, list) and v and isinstance(v[0], DC):
                        unpacked[k] = [x.data[0] for x in v]
                    else:
                        unpacked[k] = v
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **unpacked)
                results.extend(result)
                for _ in range(len(result)):
                    prog_bar.update()
            outputs = results
    else:"""

    new_content = old.sub(new_block, content)

    if new_content == content:
        print(f"⚠️  Pattern not matched in {filepath}. File may have unexpected format.")
        return False

    with open(filepath, "w") as f:
        f.write(new_content)

    print(f"✅ Patched {filepath} successfully!")
    print("   - GPU mode: used automatically on capable GPUs (compute >= 5.0)")
    print("   - CPU mode: fallback for old GPUs or CPU-only machines")
    return True


def main():
    filepath = find_test_py()
    if filepath is None:
        print("❌ Could not find BEVFormer's tools/test.py")
        print("   Run this script from the project root or from core_models/BEVFormer/")
        sys.exit(1)

    print(f"Found: {filepath}")
    patched = patch_test_py(filepath)
    if patched:
        print("\nYou can now run BEVFormer evaluation:")
        print("  cd core_models/BEVFormer")
        print("  PYTHONPATH=. python tools/test.py \\")
        print("      projects/configs/bevformer/bevformer_tiny.py \\")
        print("      ckpts/bevformer_tiny_epoch_24.pth \\")
        print("      --eval bbox")


if __name__ == "__main__":
    main()
