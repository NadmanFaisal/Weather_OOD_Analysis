#!/usr/bin/env bash
# =============================================================================
# download_weights.sh
# Downloads pretrained weights for BEVFormer and BEVFusion.
#
# Usage: bash scripts/download_weights.sh
#
# Output directory is auto-detected: uses "checkpoints/" if it exists,
# otherwise falls back to "weights/". Both are created if neither exists.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Auto-detect weights directory (checkpoints/ or weights/)
# ---------------------------------------------------------------------------
if [ -d "$REPO_ROOT/checkpoints" ]; then
    WEIGHTS_DIR="$REPO_ROOT/checkpoints"
elif [ -d "$REPO_ROOT/weights" ]; then
    WEIGHTS_DIR="$REPO_ROOT/weights"
else
    WEIGHTS_DIR="$REPO_ROOT/checkpoints"
fi

BEVFORMER_DIR="$WEIGHTS_DIR/bevformer"
BEVFUSION_DIR="$WEIGHTS_DIR/bevfusion"

mkdir -p "$BEVFORMER_DIR"
mkdir -p "$BEVFUSION_DIR"

echo ""
echo "Using weights directory: $WEIGHTS_DIR"

# ---------------------------------------------------------------------------
# Helper: download via curl only if the file does not already exist
# ---------------------------------------------------------------------------
download_if_missing() {
    local url="$1"
    local dest="$2"
    local label="$3"

    if [ -f "$dest" ]; then
        echo "[SKIP] $label already exists at $dest"
    else
        echo "[DOWNLOAD] $label → $dest"
        curl -L --progress-bar -o "$dest" "$url"
        echo "[OK] $label downloaded."
    fi
}

# ---------------------------------------------------------------------------
# BEVFormer weights  (GitHub Releases — reliable direct download)
#   bevformer_r101_dcn_24ep.pth  — nuScenes camera-only detection checkpoint
#   r101_dcn_fcos3d_pretrain.pth — ResNet-101 backbone initialisation
# Source: https://github.com/fundamentalvision/BEVFormer
# ---------------------------------------------------------------------------
echo ""
echo "=== BEVFormer Weights ==="

download_if_missing \
    "https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth" \
    "$BEVFORMER_DIR/bevformer_r101_dcn_24ep.pth" \
    "BEVFormer main checkpoint (r101, 24ep)"

download_if_missing \
    "https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth" \
    "$BEVFORMER_DIR/r101_dcn_fcos3d_pretrain.pth" \
    "BEVFormer backbone pretrain (ResNet-101 DCN)"

# ---------------------------------------------------------------------------
# BEVFusion weights
#
# bevfusion-det.pth — camera+LiDAR fusion detection checkpoint
#   Official checkpoint from the mit-han-lab/bevfusion README (nuScenes val:
#   68.52 mAP, 71.38 NDS). Hosted on Dropbox with a direct download link
#   (dl=1 forces download rather than preview). curl -L follows the redirect.
#   Source: https://github.com/mit-han-lab/bevfusion
#
# swint-nuimages-pretrained.pth — Swin-T backbone (GitHub Releases, reliable)
# ---------------------------------------------------------------------------
echo ""
echo "=== BEVFusion Weights ==="

download_if_missing \
    "https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1" \
    "$BEVFUSION_DIR/bevfusion-det.pth" \
    "BEVFusion det checkpoint (official mit-han-lab Dropbox)"

download_if_missing \
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth" \
    "$BEVFUSION_DIR/swint-nuimages-pretrained.pth" \
    "BEVFusion Swin-T backbone pretrain"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Summary: weight files present under $WEIGHTS_DIR ==="

all_present=true

for expected in \
    "$BEVFORMER_DIR/bevformer_r101_dcn_24ep.pth" \
    "$BEVFORMER_DIR/r101_dcn_fcos3d_pretrain.pth" \
    "$BEVFUSION_DIR/bevfusion-det.pth" \
    "$BEVFUSION_DIR/swint-nuimages-pretrained.pth"
do
    if [ -f "$expected" ]; then
        size=$(du -sh "$expected" 2>/dev/null | cut -f1)
        echo "  [OK $size]  $expected"
    else
        echo "  [MISSING]   $expected"
        all_present=false
    fi
done

echo ""
if [ "$all_present" = true ]; then
    echo "All weights downloaded successfully. ✓"
else
    echo "Some files are missing — check the output above for errors."
    exit 1
fi
echo ""