"""
Researcher(s): Hasan Zahid, Nadman Abdullah Bin Faisal, Vaibhav Puram

Artifact:
BEVFormer Box-Level Risk-Coverage Curve Evaluator
Methodology: Design Science Research (Cycle II: Solution Design)

Purpose:
This script evaluates the safety-filtering capability of two OOD scoring methods
(Energy Score and Mahalanobis Distance) by computing Risk-Coverage Curves at the
individual bounding box level.

Unlike the existing frame-level evaluators (energy_score.py, mahalanobis.py), this
script reads the intercepted .pt files directly and computes per-query OOD scores
without aggregation — producing one score per predicted bounding box (~90,000 per
condition across ~6,000 frames).

The Risk-Coverage Curve plots:
  - X-axis: Coverage — fraction of predicted boxes accepted by the safety monitor
  - Y-axis: Risk — error rate (FP rate) among accepted boxes
  - Boxes are ranked ascending by OOD score (most in-distribution first)
  - AURC (Area Under the Risk-Coverage Curve) summarises performance: lower = better

Two plots are produced per weather condition:
  - Raw:        Energy Score (blue)  + Raw Mahalanobis (orange)
  - Normalized: Energy Score (green) + Normalized Mahalanobis (purple)

NOTE: GT boxes in the PKL are in the LiDAR local frame while BEVFormer predictions
are in the global frame. A full LiDAR→Ego→Global coordinate transform is applied
before BEV L2 matching to ensure correct TP/FP assignment.
"""

import os
import re
import sys
import json
import pickle
import glob
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    FEATURE_LOGIT_OUTPUT, FEATURE_OUTPUT,
    RISK_COVERAGE_PLOT_OUTPUT, BEVFORMER_TEST_BASE, PKL_MAP,
)

# Minimum detection confidence — boxes below this threshold are excluded
CONFIDENCE_THRESHOLD = 0.3

# BEV L2 distance threshold in metres for TP/FP assignment
MATCH_DISTANCE_M = 2.0

# Temperature parameter for the energy score (T=1.0 is standard)
ENERGY_TEMPERATURE = 1.0

# Risk threshold for the safety annotation line on the plot
SAFETY_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# BEVFormer test directory auto-detection
# ---------------------------------------------------------------------------

def parse_bevformer_dir_ts(dir_name):
    """
    Parse a BEVFormer output directory name into a datetime object.
    Directory names use the format: Wed_May_13_19_27_11_2026
    Single-digit days appear with a double underscore (e.g. May__1),
    which is normalised to zero-padded form (May_01) before parsing.
    Returns None if the name cannot be parsed.
    """
    try:
        normalised = re.sub(r'__(\d)_', r'_0\1_', dir_name)
        return datetime.strptime(normalised, '%a_%b_%d_%H_%M_%S_%Y')
    except ValueError:
        return None


def find_bevformer_test_dir(weather, severity):
    """
    Auto-detect which BEVFormer test directory belongs to a given weather condition.

    BEVFormer evaluation runs immediately after the logit hook fires, so the test
    directory timestamp is always slightly later (~10-11 minutes) than the OOD
    logit directory timestamp. This function finds the closest BEVFormer directory
    that was created AFTER the latest OOD logit timestamp for the given condition.

    Returns the full path to the matched directory, or None if not found.
    """
    logit_base = os.path.join(FEATURE_LOGIT_OUTPUT, weather, severity)
    if not os.path.exists(logit_base):
        return None

    # Get the latest OOD logit timestamp for this condition
    ts_dirs = sorted(d for d in os.listdir(logit_base)
                     if os.path.isdir(os.path.join(logit_base, d)))
    if not ts_dirs:
        return None
    try:
        ood_dt = datetime.strptime(ts_dirs[-1], '%Y%m%d_%H%M%S')
    except ValueError:
        return None

    if not os.path.exists(BEVFORMER_TEST_BASE):
        return None

    # Find all BEVFormer test dirs that have results_nusc.json and came after ood_dt
    candidates = []
    for d in os.listdir(BEVFORMER_TEST_BASE):
        full_path = os.path.join(BEVFORMER_TEST_BASE, d)
        if not os.path.isdir(full_path):
            continue
        if not os.path.exists(os.path.join(full_path, 'pts_bbox', 'results_nusc.json')):
            continue
        dir_dt = parse_bevformer_dir_ts(d)
        if dir_dt is None:
            continue
        diff = (dir_dt - ood_dt).total_seconds()
        if diff > 0:
            candidates.append((diff, full_path))

    if not candidates:
        return None

    # Return the directory with the smallest positive time difference
    candidates.sort()
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Prediction and GT loading
# ---------------------------------------------------------------------------

def load_results_nusc(path):
    """
    Load BEVFormer predictions from results_nusc.json.
    Returns a dict mapping sample_token → list of predicted box dicts.
    Each box dict contains 'translation' [x, y, z] in global frame and 'detection_score'.
    """
    with open(path) as f:
        data = json.load(f)
    return data['results']


def quat_to_rot_matrix(q):
    """
    Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.
    Used for the LiDAR→Ego→Global coordinate transform.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2),  2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),         1 - 2*(x**2 + y**2)],
    ])


def boxes_lidar_to_global_xy(gt_boxes, sample_info):
    """
    Transform GT box centres from the LiDAR local frame to the global XY frame.

    GT annotations in the nuScenes PKL are stored in the LiDAR frame, but
    BEVFormer predictions are in the global frame. This transform is required
    to put both in the same coordinate space before BEV L2 matching.

    Transform chain:
        LiDAR → Ego:   xyz = R(lidar2ego_rotation) @ xyz + lidar2ego_translation
        Ego → Global:  xyz = R(ego2global_rotation) @ xyz + ego2global_translation

    Returns an (N, 2) array of [x, y] global coordinates. Z is discarded
    because GT matching is performed in Bird's Eye View (2D top-down).
    """
    if len(gt_boxes) == 0:
        return np.zeros((0, 2))
    xyz = gt_boxes[:, :3].copy()

    # LiDAR → Ego
    l2e_t = np.array(sample_info.get('lidar2ego_translation', [0, 0, 0]))
    l2e_r = sample_info.get('lidar2ego_rotation', [1, 0, 0, 0])
    xyz = (quat_to_rot_matrix(l2e_r) @ xyz.T).T + l2e_t

    # Ego → Global
    e2g_t = np.array(sample_info.get('ego2global_translation', [0, 0, 0]))
    e2g_r = sample_info.get('ego2global_rotation', [1, 0, 0, 0])
    xyz = (quat_to_rot_matrix(e2g_r) @ xyz.T).T + e2g_t

    return xyz[:, :2]


def load_gt_from_pkl(pkl_path):
    """
    Load ground truth boxes from a nuScenes-C PKL file and transform them
    to global XY coordinates.

    Returns a dict mapping sample_token → {'global_xy': np.ndarray of shape (N, 2)}.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data

    gt_map = {}
    for sample_info in infos:
        token = sample_info.get('token') or sample_info.get('sample_token')
        if not token:
            continue
        gt_boxes = np.array(sample_info.get('gt_boxes', []))
        gt_boxes = gt_boxes if gt_boxes.ndim == 2 else np.zeros((0, 9))
        gt_map[token] = {
            'global_xy': boxes_lidar_to_global_xy(gt_boxes, sample_info),
        }
    return gt_map


# ---------------------------------------------------------------------------
# .pt file and baseline loading
# ---------------------------------------------------------------------------

def find_logit_dir(weather, severity, timestamp):
    """
    Locate the intercepted feature logits directory for the given condition.
    Falls back to the latest available timestamp if the exact one is not found.
    """
    path = os.path.join(FEATURE_LOGIT_OUTPUT, weather, severity, timestamp)
    if os.path.exists(path):
        return path
    base = os.path.join(FEATURE_LOGIT_OUTPUT, weather, severity)
    if not os.path.exists(base):
        return None
    ts_dirs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    return os.path.join(base, ts_dirs[-1]) if ts_dirs else None


def find_baseline_timestamp():
    """
    Auto-detect the clean NuScenes baseline timestamp from the Mahalanobis
    distances folder. Returns the latest timestamp found, excluding 'normalized'.
    """
    base = os.path.join(FEATURE_OUTPUT, 'nuscenes')
    if not os.path.exists(base):
        return None
    ts_dirs = sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d != 'normalized'
    )
    return ts_dirs[-1] if ts_dirs else None


def load_raw_baseline(baseline_timestamp):
    """
    Load the raw (non-GAP) Mahalanobis baseline: mean [D] and inv_cov [D, D].

    This baseline was fitted on individual top-300 query feature vectors from
    the clean NuScenes baseline run, making it directly compatible with
    per-query scoring in this evaluator.
    """
    baseline_path = os.path.join(FEATURE_OUTPUT, 'nuscenes', baseline_timestamp, 'mahalanobis_baseline.pt')
    if not os.path.exists(baseline_path):
        print(f'[!] Raw baseline not found: {baseline_path}')
        return None, None
    data = torch.load(baseline_path, map_location='cpu')
    return data['mean'], data['inv_cov']


def load_normalized_baseline(baseline_timestamp):
    """
    Load the normalized (GAP+L2) Mahalanobis baseline: mean [D] and inv_cov [D, D].

    This baseline was fitted on GAP-averaged, L2-normalized frame-level feature
    vectors from the clean NuScenes baseline run. It is used here with L2-only
    normalization per query (GAP is skipped as it would collapse all queries into
    one frame-level vector, which is incompatible with per-query scoring).
    """
    baseline_path = os.path.join(FEATURE_OUTPUT, 'nuscenes', 'normalized', baseline_timestamp, 'mahalanobis_baseline.pt')
    if not os.path.exists(baseline_path):
        print(f'[!] Normalized baseline not found: {baseline_path}')
        return None, None
    data = torch.load(baseline_path, map_location='cpu')
    return data['mean'], data['inv_cov']


# ---------------------------------------------------------------------------
# Per-query OOD scoring
# ---------------------------------------------------------------------------

def compute_per_query_energy(logits, T=ENERGY_TEMPERATURE):
    """
    Compute the Helmholtz free energy score for every query individually.
    Returns a tensor of shape [N] — one score per query, with NO aggregation.

    Lower energy = more in-distribution (confident prediction).
    Higher energy = more out-of-distribution (uncertain/anomalous).

    logits: [1, N, C] or [N, C] where N=queries, C=classes
    """
    logits = logits.squeeze(0) if logits.dim() == 3 else logits  # [N, C]
    return -T * torch.logsumexp(logits.float() / T, dim=-1)      # [N]


def compute_per_query_mahalanobis(features, mean, inv_cov):
    """
    Compute the raw Mahalanobis distance for every query feature vector.
    Returns a tensor of shape [N] — one score per query, with NO aggregation.

    Uses the raw baseline (fitted on individual top-300 query features),
    so the feature distribution matches per-query scoring.

    features: [1, N, D] or [N, D] where N=queries, D=feature dimension (256)
    """
    features = features.squeeze(0) if features.dim() == 3 else features  # [N, D]
    delta = features.float() - mean
    return torch.sqrt((delta @ inv_cov * delta).sum(dim=1))               # [N]


def compute_per_query_mahalanobis_normalized(features, mean, inv_cov):
    """
    Compute the normalized Mahalanobis distance for every query feature vector.
    Applies L2 normalization per query (NO GAP) before scoring.
    Returns a tensor of shape [N] — one score per query, with NO aggregation.

    NOTE: GAP (Global Average Pooling) is intentionally skipped here. GAP would
    collapse all N queries into a single frame-level vector, making per-query
    scoring impossible. L2 normalization is applied per query as the closest
    approximation to the normalized baseline's feature preprocessing.

    features: [1, N, D] or [N, D]
    """
    features = features.squeeze(0) if features.dim() == 3 else features  # [N, D]
    features = F.normalize(features.float(), p=2, dim=-1)                # L2 per query, no GAP
    delta = features - mean
    return torch.sqrt((delta @ inv_cov * delta).sum(dim=1))              # [N]


def get_query_confidence(logits):
    """
    Compute the max sigmoid confidence per query.

    BEVFormer's detection_score in results_nusc.json equals max(sigmoid(logits[i]))
    for query i. Computing the same value here allows each predicted box to be
    aligned back to its originating query via nearest-neighbour matching.

    logits: [1, N, C] or [N, C] — returns tensor of shape [N]
    """
    logits = logits.squeeze(0) if logits.dim() == 3 else logits  # [N, C]
    return torch.sigmoid(logits.float()).max(dim=-1).values        # [N]


# ---------------------------------------------------------------------------
# GT matching
# ---------------------------------------------------------------------------

def match_predictions_to_gt(gt_xy, pred_xy):
    """
    Greedy BEV L2 matching between predicted boxes and GT boxes.

    Iteratively matches the closest (GT, prediction) pair within MATCH_DISTANCE_M.
    Each GT and prediction can only be matched once (one-to-one).

    Returns a bool array of shape [n_pred]:
        True  = TP (matched to a GT within threshold)
        False = FP (unmatched prediction)

    Missed GTs (FN) are not tracked — only predicted boxes are scored.
    """
    n_gt, n_pred = len(gt_xy), len(pred_xy)
    pred_matched = np.zeros(n_pred, dtype=bool)

    if n_gt == 0 or n_pred == 0:
        return pred_matched

    # Build pairwise BEV L2 distance matrix
    diffs = gt_xy[:, None, :] - pred_xy[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))  # (n_gt, n_pred)

    for _ in range(min(n_gt, n_pred)):
        if dists.min() > MATCH_DISTANCE_M:
            break
        gi, pi = np.unravel_index(dists.argmin(), dists.shape)
        pred_matched[pi] = True
        # Prevent re-matching by setting used row/column to inf
        dists[gi, :] = np.inf
        dists[:, pi] = np.inf

    return pred_matched


# ---------------------------------------------------------------------------
# Box-level (score, error) pair construction — reads .pt files directly
# ---------------------------------------------------------------------------

def build_box_pairs(gt_map, predictions_map, logit_dir,
                    mean_raw=None, inv_cov_raw=None,
                    mean_norm=None, inv_cov_norm=None):
    """
    Build per-box (OOD score, error flag) pairs for all frames in logit_dir.

    For each frame (.pt file):
      1. Load intercepted logits [1, 900, 10] and features [1, 900, 256]
      2. Compute per-query OOD scores (energy, raw Maha, normalized Maha)
      3. Filter predicted boxes by CONFIDENCE_THRESHOLD
      4. Match predicted boxes to GT using greedy BEV L2 — assign TP/FP error flags
      5. Align each predicted box to its query via detection_score matching:
         BEVFormer's detection_score = max(sigmoid(logits[i])) for query i,
         so argmin(|conf - detection_score|) recovers the originating query index
      6. Append (score, error) to the respective output list

    Returns three flat lists (one entry per predicted box across all frames):
      energy_pairs    : [(energy_score, error_flag), ...]
      maha_raw_pairs  : [(maha_score,   error_flag), ...]  — raw Mahalanobis
      maha_norm_pairs : [(maha_score,   error_flag), ...]  — normalized Mahalanobis
    """
    pt_files = glob.glob(os.path.join(logit_dir, '*.pt'))
    if not pt_files:
        print(f'[!] No .pt files found in: {logit_dir}')
        return [], [], []

    energy_pairs    = []
    maha_raw_pairs  = []
    maha_norm_pairs = []
    total = len(pt_files)

    for idx, file_path in enumerate(pt_files):
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f'  [{idx+1}/{total}] frames processed...', flush=True)
        try:
            pt_data = torch.load(file_path, map_location='cpu')
            token = pt_data['sample_token']

            if token not in gt_map:
                continue

            logits   = pt_data['logits']    # [1, 900, 10] — raw classification logits
            features = pt_data['features']  # [1, 900, 256] — query feature embeddings

            # Compute per-query OOD scores — no aggregation, one score per query
            query_energies  = compute_per_query_energy(logits)       # [900]
            query_conf      = get_query_confidence(logits)           # [900] — for box alignment

            query_maha_raw  = None
            query_maha_norm = None
            if mean_raw is not None and inv_cov_raw is not None:
                query_maha_raw  = compute_per_query_mahalanobis(features, mean_raw, inv_cov_raw)
            if mean_norm is not None and inv_cov_norm is not None:
                query_maha_norm = compute_per_query_mahalanobis_normalized(features, mean_norm, inv_cov_norm)

            # Filter predicted boxes by confidence threshold
            preds = [p for p in predictions_map.get(token, [])
                     if p['detection_score'] >= CONFIDENCE_THRESHOLD]
            if not preds:
                continue

            # Assign TP/FP error flags via greedy BEV L2 matching against GT
            gt_xy   = gt_map[token]['global_xy']
            pred_xy = np.array([[p['translation'][0], p['translation'][1]] for p in preds])
            pred_matched = match_predictions_to_gt(gt_xy, pred_xy)

            # Align each predicted box to its originating query
            conf_np = query_conf.numpy()

            for i, pred in enumerate(preds):
                best_idx = int(np.abs(conf_np - pred['detection_score']).argmin())
                error = 0 if pred_matched[i] else 1  # 0 = TP, 1 = FP

                energy_pairs.append((float(query_energies[best_idx].item()), error))
                if query_maha_raw is not None:
                    maha_raw_pairs.append((float(query_maha_raw[best_idx].item()), error))
                if query_maha_norm is not None:
                    maha_norm_pairs.append((float(query_maha_norm[best_idx].item()), error))

        except Exception as e:
            print(f'[!] Failed to process {os.path.basename(file_path)}: {e}')

    return energy_pairs, maha_raw_pairs, maha_norm_pairs


# ---------------------------------------------------------------------------
# Risk-Coverage computation
# ---------------------------------------------------------------------------

def compute_risk_coverage(pairs):
    """
    Compute the Risk-Coverage curve from a list of (OOD score, error flag) pairs.

    Algorithm:
      1. Sort pairs ascending by OOD score — most in-distribution (confident) first
      2. coverage[i] = (i+1) / N  — fraction of boxes accepted up to index i
      3. risk[i] = cumulative mean of error flags up to index i
                 = running FP rate as more boxes are accepted
      4. AURC = trapezoidal area under the risk-coverage curve

    Lower AURC = better OOD discriminator (risk stays low across more coverage).
    A random discriminator gives AURC ≈ overall FP rate.

    Returns (coverage, risk, aurc) arrays, or (None, None, None) if input is empty.
    """
    if not pairs:
        return None, None, None

    scores = np.array([p[0] for p in pairs])
    errors = np.array([p[1] for p in pairs], dtype=float)

    # Sort ascending — lowest OOD score (most in-distribution) accepted first
    sorted_idx    = np.argsort(scores)
    errors_sorted = errors[sorted_idx]

    n        = len(errors_sorted)
    coverage = np.arange(1, n + 1) / n
    risk     = np.cumsum(errors_sorted) / np.arange(1, n + 1)
    aurc     = float(np.trapz(risk, coverage))

    return coverage, risk, aurc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_risk_coverage(curve_data, weather, severity, save_dir, normalized=False):
    """
    Plot the Risk-Coverage Curve for a given weather condition.

    Draws one line per OOD metric. For the Mahalanobis Distance curve, annotates
    the point of maximum coverage that still satisfies the 5% risk safety threshold.

    Color scheme:
      Raw plot:        Energy Score = cornflowerblue,  Mahalanobis = darkorange
      Normalized plot: Energy Score = mediumseagreen,  Mahalanobis = mediumpurple

    Saves to save_dir/risk_coverage_{weather}_{severity}.png at 300 dpi.
    """
    plt.figure(figsize=(8, 6))

    # Raw plot uses blue/orange; normalized plot uses green/purple for visual distinction
    if normalized:
        colors = {
            'Energy Score':         'mediumseagreen',
            'Mahalanobis Distance': 'mediumpurple',
        }
    else:
        colors = {
            'Energy Score':         'cornflowerblue',
            'Mahalanobis Distance': 'darkorange',
        }

    for metric_name, data in curve_data.items():
        coverage = data['coverage']
        risk     = data['risk']
        color    = colors.get(metric_name, 'gray')

        plt.plot(
            coverage, risk,
            color=color, lw=2,
            label=f'{metric_name} (AURC = {data["aurc"]:.4f})',
        )

        # Annotate the maximum safe coverage at the 5% risk threshold.
        # Only applied to Mahalanobis Distance (not Energy Score).
        # Skipped if the curve never meaningfully drops below the threshold
        # (safe_cov < 0.5% is treated as threshold never reached).
        if metric_name == 'Mahalanobis Distance':
            indices_below = np.where(risk <= SAFETY_THRESHOLD)[0]
            if len(indices_below) > 0:
                idx       = indices_below[-1]   # last index still at/below threshold
                safe_cov  = coverage[idx]
                safe_risk = risk[idx]
                if safe_cov >= 0.005:
                    plt.plot(safe_cov, safe_risk, 'o', color=color, markersize=8, zorder=5)
                    plt.text(
                        safe_cov + 0.01, safe_risk,
                        f'{safe_cov * 100:.1f}% Coverage',
                        color=color, fontsize=9, fontweight='bold', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8),
                    )

    # Horizontal dashed line marking the 5% risk safety target
    plt.axhline(y=SAFETY_THRESHOLD, color='red', linestyle='--', lw=1.5, label='5% Risk Target')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Coverage (Proportion of Accepted Predictions)')
    plt.ylabel('Risk (Error Rate)')
    variant = ' (Normalized)' if normalized else ''
    plt.title(f'Risk-Coverage Curve{variant}: {weather.capitalize()} ({severity.capitalize()})')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'risk_coverage_{weather}_{severity}.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f'Risk-Coverage Curve saved to: {file_path}')
    plt.close()


# ---------------------------------------------------------------------------
# Per-condition runner
# ---------------------------------------------------------------------------

def get_latest_timestamp(weather, severity):
    """Return the latest intercepted logit timestamp for a given condition."""
    base = os.path.join(FEATURE_LOGIT_OUTPUT, weather, severity)
    if not os.path.exists(base):
        return None
    ts_dirs = sorted(d for d in os.listdir(base)
                     if os.path.isdir(os.path.join(base, d)))
    return ts_dirs[-1] if ts_dirs else None


def run_condition(weather, severity, timestamp):
    """
    Run the full Risk-Coverage evaluation pipeline for one weather condition.

    Steps:
      1. Locate the BEVFormer predictions (results_nusc.json) via timestamp matching
      2. Load GT from PKL and transform coordinates to global frame
      3. Load Mahalanobis baselines (raw and normalized)
      4. Stream .pt files and build per-box (score, error) pairs
      5. Compute Risk-Coverage curves and AURC for each metric
      6. Save raw plot  → plots/risk_coverage/{weather}/{severity}/{timestamp}/
         Save norm plot → plots/risk_coverage/{weather}/{severity}/normalized/{timestamp}/
    """
    print(f'\n========================================================='
          f'\nProcessing: {weather} ({severity}) | timestamp: {timestamp}'
          f'\n=========================================================')

    # --- BEVFormer predictions ---
    bevformer_test_dir = os.environ.get('BEVFORMER_TEST_DIR') or find_bevformer_test_dir(weather, severity)
    if not bevformer_test_dir:
        print(f'[!] Could not find BEVFormer test dir for ({weather}, {severity}). Skipping.')
        return

    results_nusc_path = os.path.join(bevformer_test_dir, 'pts_bbox', 'results_nusc.json')
    if not os.path.exists(results_nusc_path):
        print(f'[!] results_nusc.json not found: {results_nusc_path}. Skipping.')
        return

    # --- Ground truth ---
    pkl_path = PKL_MAP.get((weather, severity))
    if not pkl_path or not os.path.exists(pkl_path):
        print(f'[!] Ground truth PKL not found: {pkl_path}. Skipping.')
        return

    print(f'Loading predictions : {results_nusc_path}')
    predictions_map = load_results_nusc(results_nusc_path)

    print(f'Loading ground truth: {pkl_path}')
    gt_map = load_gt_from_pkl(pkl_path)

    # Sanity check: verify GT and prediction coordinates are in the same frame.
    # Distance should be < 50m for a correctly matched first GT/prediction pair.
    for token, gt in gt_map.items():
        preds = predictions_map.get(token, [])
        if len(gt['global_xy']) > 0 and preds:
            gt_xy   = gt['global_xy'][0]
            pred_xy = preds[0]['translation'][:2]
            dist    = np.linalg.norm(gt_xy - np.array(pred_xy))
            print(f'[sanity] First GT global xy  : {gt_xy}')
            print(f'[sanity] First pred global xy: {pred_xy}')
            print(f'[sanity] Distance            : {dist:.1f}m  (should be <50m if coords match)')
            break

    # --- Intercepted .pt files ---
    logit_dir = find_logit_dir(weather, severity, timestamp)
    if not logit_dir:
        print(f'[!] Could not find intercepted logits for ({weather}, {severity}, {timestamp}). Skipping.')
        return
    print(f'Intercepted logits  : {logit_dir}')

    # --- Mahalanobis baselines ---
    baseline_ts = os.environ.get('BASELINE_TIMESTAMP') or find_baseline_timestamp()
    mean_raw, inv_cov_raw   = None, None
    mean_norm, inv_cov_norm = None, None
    if baseline_ts:
        mean_raw, inv_cov_raw = load_raw_baseline(baseline_ts)
        if mean_raw is not None:
            print(f'Raw baseline loaded        : nuscenes/{baseline_ts}')
        mean_norm, inv_cov_norm = load_normalized_baseline(baseline_ts)
        if mean_norm is not None:
            print(f'Normalized baseline loaded : nuscenes/normalized/{baseline_ts}')
    else:
        print('[!] No baseline timestamp found — Mahalanobis curves will be skipped.')

    # --- Build per-box (score, error_flag) pairs from .pt files ---
    print('Building per-box pairs from .pt files...')
    energy_pairs, maha_raw_pairs, maha_norm_pairs = build_box_pairs(
        gt_map, predictions_map, logit_dir,
        mean_raw, inv_cov_raw, mean_norm, inv_cov_norm,
    )

    # --- Raw plot: Energy Score + Raw Mahalanobis ---
    curve_data_raw = {}

    if energy_pairs:
        n_boxes = len(energy_pairs)
        n_fp    = sum(e for _, e in energy_pairs)
        print(f'Energy Score        -> {n_boxes} boxes | FP: {n_fp} ({100*n_fp/n_boxes:.1f}%)')
        coverage, risk, aurc = compute_risk_coverage(energy_pairs)
        if coverage is not None:
            curve_data_raw['Energy Score'] = {'coverage': coverage, 'risk': risk, 'aurc': aurc}
            print(f'  AURC: {aurc:.4f}')
    else:
        print('[!] No energy pairs built — check .pt files.')

    if maha_raw_pairs:
        n_boxes = len(maha_raw_pairs)
        n_fp    = sum(e for _, e in maha_raw_pairs)
        print(f'Mahalanobis (Raw)   -> {n_boxes} boxes | FP: {n_fp} ({100*n_fp/n_boxes:.1f}%)')
        coverage, risk, aurc = compute_risk_coverage(maha_raw_pairs)
        if coverage is not None:
            curve_data_raw['Mahalanobis Distance'] = {'coverage': coverage, 'risk': risk, 'aurc': aurc}
            print(f'  AURC: {aurc:.4f}')
    else:
        print('[!] No raw Mahalanobis pairs built — check baseline or .pt files.')

    if curve_data_raw:
        save_dir = os.path.join(RISK_COVERAGE_PLOT_OUTPUT, weather, severity, timestamp)
        plot_risk_coverage(curve_data_raw, weather, severity, save_dir, normalized=False)

    # --- Normalized plot: Energy Score + Normalized Mahalanobis ---
    # Only saved if normalized Mahalanobis data is available.
    # Energy Score is reused from the raw computation (no normalization variant).
    curve_data_norm = {}

    if energy_pairs:
        curve_data_norm['Energy Score'] = curve_data_raw.get('Energy Score')

    if maha_norm_pairs:
        n_boxes = len(maha_norm_pairs)
        n_fp    = sum(e for _, e in maha_norm_pairs)
        print(f'Mahalanobis (Norm)  -> {n_boxes} boxes | FP: {n_fp} ({100*n_fp/n_boxes:.1f}%)')
        coverage, risk, aurc = compute_risk_coverage(maha_norm_pairs)
        if coverage is not None:
            curve_data_norm['Mahalanobis Distance'] = {'coverage': coverage, 'risk': risk, 'aurc': aurc}
            print(f'  AURC (norm): {aurc:.4f}')
    else:
        print('[!] No normalized Mahalanobis pairs built — check normalized baseline or .pt files.')

    curve_data_norm = {k: v for k, v in curve_data_norm.items() if v is not None}
    if 'Mahalanobis Distance' in curve_data_norm:
        save_dir_norm = os.path.join(RISK_COVERAGE_PLOT_OUTPUT, weather, severity, 'normalized', timestamp)
        plot_risk_coverage(curve_data_norm, weather, severity, save_dir_norm, normalized=True)

    print('\n---------------- FINAL AURC ---------------------')
    if not curve_data_raw and not curve_data_norm:
        print('No valid scores found. Check paths above.')
    else:
        print('Raw:')
        for metric, data in curve_data_raw.items():
            print(f'  {metric}: AURC = {data["aurc"]:.4f}')
        print('Normalized:')
        for metric, data in curve_data_norm.items():
            print(f'  {metric}: AURC = {data["aurc"]:.4f}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # If specific condition env vars are set, run only that condition.
    # Otherwise auto-loop all 6 conditions using the latest available timestamp.
    weather   = os.environ.get('OOD_WEATHER')
    severity  = os.environ.get('OOD_SEVERITY')
    timestamp = os.environ.get('OOD_TIMESTAMP')

    if weather and severity and timestamp:
        run_condition(weather, severity, timestamp)
    else:
        for w, s in PKL_MAP.keys():
            ts = get_latest_timestamp(w, s)
            if ts:
                run_condition(w, s, ts)
            else:
                print(f'[!] No intercepted logits found for ({w}, {s}). Skipping.')
