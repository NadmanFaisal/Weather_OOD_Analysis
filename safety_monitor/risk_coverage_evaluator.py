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
    RISK_COVERAGE_PLOT_OUTPUT, BEVFORMER_TEST_BASE,
)

CONFIDENCE_THRESHOLD = 0.3
MATCH_DISTANCE_M = 2.0
ENERGY_TEMPERATURE = 1.0

PKL_MAP = {
    ('Fog',  'easy'): 'data/sets/nuscenes-c/nuScenes-c/Fog/easy/nuscenes_infos_temporal_val.pkl',
    ('Fog',  'mid'):  'data/sets/nuscenes-c/nuScenes-c/Fog/mid/nuscenes_infos_temporal_val.pkl',
    ('Fog',  'hard'): 'data/sets/nuscenes-c/nuScenes-c/Fog/hard/nuscenes_infos_temporal_val.pkl',
    ('Snow', 'easy'): 'data/sets/nuscenes-c/nuScenes-c/Snow/easy/nuscenes_infos_temporal_val.pkl',
    ('Snow', 'mid'):  'data/sets/nuscenes-c/nuScenes-c/Snow/mid/nuscenes_infos_temporal_val.pkl',
    ('Snow', 'hard'): 'data/sets/nuscenes-c/nuScenes-c/Snow/hard/nuscenes_infos_temporal_val.pkl',
}


# ---------------------------------------------------------------------------
# BEVFormer test directory auto-detection (unchanged)
# ---------------------------------------------------------------------------

def parse_bevformer_dir_ts(dir_name):
    try:
        normalised = re.sub(r'__(\d)_', r'_0\1_', dir_name)
        return datetime.strptime(normalised, '%a_%b_%d_%H_%M_%S_%Y')
    except ValueError:
        return None


def find_bevformer_test_dir(weather, severity):
    logit_base = os.path.join(FEATURE_LOGIT_OUTPUT, weather, severity)
    if not os.path.exists(logit_base):
        return None
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
    candidates.sort()
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Prediction and GT loading
# ---------------------------------------------------------------------------

def load_results_nusc(path):
    with open(path) as f:
        data = json.load(f)
    return data['results']


def quat_to_rot_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2),  2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),         1 - 2*(x**2 + y**2)],
    ])


def boxes_lidar_to_global_xy(gt_boxes, sample_info):
    if len(gt_boxes) == 0:
        return np.zeros((0, 2))
    xyz = gt_boxes[:, :3].copy()
    l2e_t = np.array(sample_info.get('lidar2ego_translation', [0, 0, 0]))
    l2e_r = sample_info.get('lidar2ego_rotation', [1, 0, 0, 0])
    xyz = (quat_to_rot_matrix(l2e_r) @ xyz.T).T + l2e_t
    e2g_t = np.array(sample_info.get('ego2global_translation', [0, 0, 0]))
    e2g_r = sample_info.get('ego2global_rotation', [1, 0, 0, 0])
    xyz = (quat_to_rot_matrix(e2g_r) @ xyz.T).T + e2g_t
    return xyz[:, :2]


def load_gt_from_pkl(pkl_path):
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
    """Locate the intercepted feature logits directory for this condition."""
    path = os.path.join(FEATURE_LOGIT_OUTPUT, weather, severity, timestamp)
    if os.path.exists(path):
        return path
    # Fall back to latest available
    base = os.path.join(FEATURE_LOGIT_OUTPUT, weather, severity)
    if not os.path.exists(base):
        return None
    ts_dirs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    return os.path.join(base, ts_dirs[-1]) if ts_dirs else None


def find_baseline_timestamp():
    """Auto-detect the clean baseline timestamp from the mahalanobis distances folder."""
    base = os.path.join(FEATURE_OUTPUT, 'nuscenes')
    if not os.path.exists(base):
        return None
    ts_dirs = sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d != 'normalized'
    )
    return ts_dirs[-1] if ts_dirs else None


def load_raw_baseline(baseline_timestamp):
    """Load the raw (non-GAP) Mahalanobis baseline fitted on individual top-300 query features."""
    baseline_path = os.path.join(FEATURE_OUTPUT, 'nuscenes', baseline_timestamp, 'mahalanobis_baseline.pt')
    if not os.path.exists(baseline_path):
        print(f'[!] Raw baseline not found: {baseline_path}')
        return None, None
    data = torch.load(baseline_path, map_location='cpu')
    return data['mean'], data['inv_cov']


def load_normalized_baseline(baseline_timestamp):
    """Load the normalized (GAP+L2) Mahalanobis baseline fitted on frame-level features."""
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
    """Energy score for every query. Returns tensor of shape [N].
    Lower energy = more in-distribution (confident). Higher = OOD.
    logits: [1, N, C] or [N, C]
    """
    logits = logits.squeeze(0) if logits.dim() == 3 else logits  # [N, C]
    return -T * torch.logsumexp(logits.float() / T, dim=-1)      # [N]


def compute_per_query_mahalanobis(features, mean, inv_cov):
    """Mahalanobis distance for every query feature vector. Returns tensor of shape [N].
    features: [1, N, D] or [N, D]
    """
    features = features.squeeze(0) if features.dim() == 3 else features  # [N, D]
    delta = features.float() - mean
    return torch.sqrt((delta @ inv_cov * delta).sum(dim=1))               # [N]


def compute_per_query_mahalanobis_normalized(features, mean, inv_cov):
    """L2-normalize each query feature (no GAP), then compute Mahalanobis distance.
    Uses the normalized baseline (fitted on GAP+L2 frame vectors) but applied per-query.
    features: [1, N, D] or [N, D]
    """
    features = features.squeeze(0) if features.dim() == 3 else features  # [N, D]
    features = F.normalize(features.float(), p=2, dim=-1)                # L2 per query
    delta = features - mean
    return torch.sqrt((delta @ inv_cov * delta).sum(dim=1))              # [N]


def get_query_confidence(logits):
    """Max sigmoid confidence per query — used to align queries with predicted boxes.
    logits: [1, N, C] or [N, C] — returns tensor of shape [N].
    """
    logits = logits.squeeze(0) if logits.dim() == 3 else logits  # [N, C]
    return torch.sigmoid(logits.float()).max(dim=-1).values        # [N]


# ---------------------------------------------------------------------------
# GT matching
# ---------------------------------------------------------------------------

def match_predictions_to_gt(gt_xy, pred_xy):
    """Greedy BEV L2 matching. Returns bool array [n_pred]: True = TP, False = FP."""
    n_gt, n_pred = len(gt_xy), len(pred_xy)
    pred_matched = np.zeros(n_pred, dtype=bool)

    if n_gt == 0 or n_pred == 0:
        return pred_matched

    diffs = gt_xy[:, None, :] - pred_xy[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))  # (n_gt, n_pred)

    for _ in range(min(n_gt, n_pred)):
        if dists.min() > MATCH_DISTANCE_M:
            break
        gi, pi = np.unravel_index(dists.argmin(), dists.shape)
        pred_matched[pi] = True
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
    Reads each .pt file (one per frame), computes per-query OOD scores,
    aligns each query to its predicted bounding box by confidence score matching,
    then assigns a binary error flag (0=TP, 1=FP) via GT matching.

    Returns three flat lists:
      energy_pairs    : [(energy_score, error_flag), ...]  — one per predicted box
      maha_raw_pairs  : [(maha_score,   error_flag), ...]  — raw Mahalanobis
      maha_norm_pairs : [(maha_score,   error_flag), ...]  — normalized Mahalanobis (L2 features)
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

            logits   = pt_data['logits']    # [1, N, C]
            features = pt_data['features']  # [1, N, D]

            # Per-query OOD scores
            query_energies  = compute_per_query_energy(logits)       # [N]
            query_conf      = get_query_confidence(logits)           # [N]  ← used for alignment

            query_maha_raw  = None
            query_maha_norm = None
            if mean_raw is not None and inv_cov_raw is not None:
                query_maha_raw  = compute_per_query_mahalanobis(features, mean_raw, inv_cov_raw)            # [N]
            if mean_norm is not None and inv_cov_norm is not None:
                query_maha_norm = compute_per_query_mahalanobis_normalized(features, mean_norm, inv_cov_norm)  # [N]

            # Predicted boxes for this frame (filtered by confidence)
            preds = [p for p in predictions_map.get(token, [])
                     if p['detection_score'] >= CONFIDENCE_THRESHOLD]
            if not preds:
                continue

            # GT matching → per-box error flags
            gt_xy   = gt_map[token]['global_xy']
            pred_xy = np.array([[p['translation'][0], p['translation'][1]] for p in preds])
            pred_matched = match_predictions_to_gt(gt_xy, pred_xy)

            # Align each predicted box to its query by detection score matching.
            # BEVFormer's detection_score = max(sigmoid(logits[i])) for query i,
            # so matching on this value gives us the exact corresponding query.
            conf_np = query_conf.numpy()

            for i, pred in enumerate(preds):
                best_idx = int(np.abs(conf_np - pred['detection_score']).argmin())
                error = 0 if pred_matched[i] else 1

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
    Sort pairs ascending by OOD score (most confident first).
    coverage[i] = (i+1) / N
    risk[i]     = mean(errors[0..i])
    AURC        = area under the curve via trapezoidal rule.
    """
    if not pairs:
        return None, None, None

    scores = np.array([p[0] for p in pairs])
    errors = np.array([p[1] for p in pairs], dtype=float)

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

SAFETY_THRESHOLD = 0.05

def plot_risk_coverage(curve_data, weather, severity, save_dir, normalized=False):
    plt.figure(figsize=(8, 6))

    # Raw plot: blue + orange.  Normalized plot: green + purple.
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

        # Safety threshold intersection — Mahalanobis only, and only if coverage is meaningful
        if metric_name == 'Mahalanobis Distance':
            indices_below = np.where(risk <= SAFETY_THRESHOLD)[0]
            if len(indices_below) > 0:
                idx       = indices_below[-1]
                safe_cov  = coverage[idx]
                safe_risk = risk[idx]
                if safe_cov >= 0.005:  # ignore near-zero coverage (threshold never meaningfully reached)
                    plt.plot(safe_cov, safe_risk, 'o', color=color, markersize=8, zorder=5)
                    plt.text(
                        safe_cov + 0.01, safe_risk,
                        f'{safe_cov * 100:.1f}% Coverage',
                        color=color, fontsize=9, fontweight='bold', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8),
                    )

    # 5% safety threshold line
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
    base = os.path.join(FEATURE_LOGIT_OUTPUT, weather, severity)
    if not os.path.exists(base):
        return None
    ts_dirs = sorted(d for d in os.listdir(base)
                     if os.path.isdir(os.path.join(base, d)))
    return ts_dirs[-1] if ts_dirs else None


def run_condition(weather, severity, timestamp):
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

    # Coordinate sanity check
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

    # --- Baselines ---
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

    # --- Build per-box (score, error_flag) pairs ---
    print('Building per-box pairs from .pt files...')
    energy_pairs, maha_raw_pairs, maha_norm_pairs = build_box_pairs(
        gt_map, predictions_map, logit_dir,
        mean_raw, inv_cov_raw, mean_norm, inv_cov_norm,
    )

    # --- Raw plot ---
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

    # --- Normalized plot ---
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
