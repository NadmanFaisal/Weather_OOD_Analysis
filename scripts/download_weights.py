"""
Artifact:
nuScenes Data Ingestion & Integrity Pipeline
Methodology: Design Science Research (Cycle II: Solution Design)
Researcher(s): Hasan Zahid, Nadman Abdullah Bin Faisal, Vaibhav Puram
 
download_weights.py
===================
Purpose: Downloads pretrained weights for BEVFormer and BEVFusion.
 
Usage:
    python scripts/download_weights.py
 
Works on macOS (Apple Silicon), Linux, and Windows.
Requires only Python standard library — no pip installs needed.
 
Output directory is auto-detected: uses checkpoints/ if it exists,
otherwise falls back to weights/. Created automatically if neither exists.
 
---------------------------------------------------------------------------
Weight Sources
---------------------------------------------------------------------------
BEVFormer (Li et al., ECCV 2022 — https://arxiv.org/abs/2203.17270)
  Repo   : https://github.com/fundamentalvision/BEVFormer
  Weights: https://github.com/zhiqi-li/storage/releases/tag/v1.0
    - bevformer_r101_dcn_24ep.pth   : main nuScenes detection checkpoint
    - r101_dcn_fcos3d_pretrain.pth  : ResNet-101 backbone initialisation
 
BEVFusion (Liu et al., ICRA 2023 — https://arxiv.org/abs/2205.13542)
  Repo   : https://github.com/mit-han-lab/bevfusion (archived Jan 2025)
  Weights:
    - bevfusion-det.pth             : camera+LiDAR detection checkpoint
                                      (68.52 mAP / 71.38 NDS on nuScenes val)
                                      hosted on official Dropbox in README
    - swint-nuimages-pretrained.pth : Swin-T backbone (Liu et al., ICCV 2021)
                                      https://github.com/SwinTransformer/storage
---------------------------------------------------------------------------
"""

import os
import sys
import urllib.request
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Weight definitions: (url, relative_dest, label)
# ---------------------------------------------------------------------------
WEIGHTS = [
    # --- BEVFormer (GitHub Releases — reliable direct download) ---
    # Source: https://github.com/fundamentalvision/BEVFormer
    (
        "https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth",
        "bevformer/bevformer_r101_dcn_24ep.pth",
        "BEVFormer main checkpoint (r101, 24ep)",
    ),
    (
        "https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth",
        "bevformer/r101_dcn_fcos3d_pretrain.pth",
        "BEVFormer backbone pretrain (ResNet-101 DCN)",
    ),
    # --- BEVFusion ---
    # Official checkpoint from mit-han-lab/bevfusion README
    # nuScenes val: 68.52 mAP, 71.38 NDS
    # Source: https://github.com/mit-han-lab/bevfusion
    (
        "https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1",
        "bevfusion/bevfusion-det.pth",
        "BEVFusion det checkpoint (official mit-han-lab Dropbox)",
    ),
    # Swin-T backbone pretrain (GitHub Releases — reliable direct download)
    (
        "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
        "bevfusion/swint-nuimages-pretrained.pth",
        "BEVFusion Swin-T backbone pretrain",
    ),
]


# ---------------------------------------------------------------------------
# Progress bar shown during download
# ---------------------------------------------------------------------------
def _make_progress_hook(label: str):
    last_pct = [-1]

    def hook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(int(downloaded * 100 / total_size), 100)
        if pct != last_pct[0]:
            bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
            mb_done = downloaded / 1_048_576
            mb_total = total_size / 1_048_576
            print(
                f"\r  [{bar}] {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MB",
                end="",
                flush=True,
            )
            last_pct[0] = pct
        if pct == 100:
            print()  # newline after completion

    return hook


# ---------------------------------------------------------------------------
# Download a single file, skipping if already present and correct size
# ---------------------------------------------------------------------------
def download_if_missing(url: str, dest: Path, label: str) -> bool:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  [SKIP] Already exists: {dest.name}")
        return True

    print(f"  [DOWNLOAD] {label}")
    print(f"             → {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_make_progress_hook(label))
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1_048_576
        print(f"  [OK] Downloaded {size_mb:.1f} MB")
        return True
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"\n  [FAIL] {e}")
        return False


# ---------------------------------------------------------------------------
# Auto-detect the weights root directory
# ---------------------------------------------------------------------------
def find_weights_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    for candidate in ["checkpoints", "weights"]:
        d = repo_root / candidate
        if d.is_dir():
            return d

    # Neither exists — default to checkpoints/
    default = repo_root / "checkpoints"
    default.mkdir(parents=True, exist_ok=True)
    return default


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    weights_dir = find_weights_dir()
    print(f"\nUsing weights directory: {weights_dir}\n")

    sections = {
        "BEVFormer": [w for w in WEIGHTS if w[1].startswith("bevformer")],
        "BEVFusion": [w for w in WEIGHTS if w[1].startswith("bevfusion")],
    }

    results = {}
    for section, entries in sections.items():
        print(f"=== {section} Weights ===")
        for url, rel_dest, label in entries:
            dest = weights_dir / rel_dest
            success = download_if_missing(url, dest, label)
            results[rel_dest] = (dest, success)
        print()

    # --- Summary ---
    print(f"=== Summary: weight files under {weights_dir} ===")
    all_ok = True
    for rel_dest, (dest, success) in results.items():
        if dest.exists() and dest.stat().st_size > 1_000_000:
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  [OK {size_mb:>6.0f}M]  {dest}")
        else:
            print(f"  [MISSING]         {dest}")
            all_ok = False

    print()
    if all_ok:
        print("All weights downloaded successfully. ✓")
        sys.exit(0)
    else:
        print("Some files are missing — check the output above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()