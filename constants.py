# NUSCENES contstants
OUTPUT_DIR = "data/sets/nuscenes/"
REGION = 'us'
FILES = {
    # Include the meta files at the top
    # "v1.0-test_meta.tgz":"b0263f5c41b780a5a10ede2da99539eb",
    # "v1.0-trainval_meta.tgz":"537d3954ec34e5bcb89a35d4f6fb0d4a",

    # Define the blobs
    "v1.0-mini.tgz":"dc2267f6e62b221efb9575929028603b",
    #"v1.0-test_blobs.tgz":"e065445b6019ecc15c70ad9d99c47b33",
    #"v1.0-trainval01_blobs.tgz":"cbf32d2ea6996fc599b32f724e7ce8f2",
    #"v1.0-trainval02_blobs.tgz":"aeecea4878ec3831d316b382bb2f72da",
    #"v1.0-trainval03_blobs.tgz":"595c29528351060f94c935e3aaf7b995",
    #"v1.0-trainval04_blobs.tgz":"b55eae9b4aa786b478858a3fc92fb72d",
    #"v1.0-trainval05_blobs.tgz":"1c815ed607a11be7446dcd4ba0e71ed0",
    #"v1.0-trainval06_blobs.tgz":"7273eeea36e712be290472859063a678",
    #"v1.0-trainval07_blobs.tgz":"46674d2b2b852b7a857d2c9a87fc755f",
    #"v1.0-trainval08_blobs.tgz":"37524bd4edee2ab99678909334313adf",
    #"v1.0-trainval09_blobs.tgz":"a7fcd6d9c0934e4052005aa0b84615c0",
    #"v1.0-trainval10_blobs.tgz":"31e795f2c13f62533c727119b822d739",
}

CORE_NUSCENES_FOLDER = ["maps", "sweeps", "v1.0-trainval"]
LIDAR_SENSORS = [
    "LIDAR_TOP", "RADAR_FRONT", "RADAR_FRONT_LEFT", 
    "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"
]

NUSCENES_CAMERAS = [
    'CAM_FRONT', 
    'CAM_FRONT_RIGHT', 
    'CAM_FRONT_LEFT', 
    'CAM_BACK', 
    'CAM_BACK_LEFT', 
    'CAM_BACK_RIGHT', 
]

# Image corruption types
CORRUPTION_MAP = {
    'fog': 'fog',
    'snow': 'snow',
    'rain': 'spatter',
    'sun_glare': 'brightness'
}

# Model Weights
WEIGHTS = {
    "bevformer/bevformer_r101_dcn_24ep.pth": {
        "url": "https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth",
        "label": "BEVFormer main checkpoint (r101, 24ep)",
    },
    "bevformer/r101_dcn_fcos3d_pretrain.pth": {
        "url": "https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth",
        "label": "BEVFormer backbone pretrain (ResNet-101 DCN)",
    },
    "bevfusion/bevfusion-det.pth": {
        "url": "https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1",
        "label": "BEVFusion det checkpoint (official mit-han-lab Dropbox)",
    },
    "bevfusion/swint-nuimages-pretrained.pth": {
        "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
        "label": "BEVFusion Swin-T backbone pretrain",
    },
}

# PKL files
CLEAN_PKL = 'data/sets/nuscenes/nuscenes_infos_temporal_val.pkl'

OUTPUT_PKL_PATHS = [
    #Fog
    'data/sets/nuscenes-c/nuScenes-c/Fog/easy/nuscenes_infos_temporal_val.pkl',
    'data/sets/nuscenes-c/nuScenes-c/Fog/mid/nuscenes_infos_temporal_val.pkl',
    'data/sets/nuscenes-c/nuScenes-c/Fog/hard/nuscenes_infos_temporal_val.pkl',

    #Snow
    'data/sets/nuscenes-c/nuScenes-c/Snow/easy/nuscenes_infos_temporal_val.pkl',
    'data/sets/nuscenes-c/nuScenes-c/Snow/mid/nuscenes_infos_temporal_val.pkl',
    'data/sets/nuscenes-c/nuScenes-c/Snow/hard/nuscenes_infos_temporal_val.pkl',
]

AUTOMOLD_OUTPUT_PKL_PATHS = [
    #Fog
    'data/sets/nuscenes-automold/Fog/easy/nuscenes_infos_temporal_val.pkl',
    'data/sets/nuscenes-automold/Fog/mid/nuscenes_infos_temporal_val.pkl',
    'data/sets/nuscenes-automold/Fog/hard/nuscenes_infos_temporal_val.pkl',

    #Snow
    'data/sets/nuscenes-automold/Snow/easy/nuscenes_infos_temporal_val.pkl',
    'data/sets/nuscenes-automold/Snow/mid/nuscenes_infos_temporal_val.pkl',
    'data/sets/nuscenes-automold/Snow/hard/nuscenes_infos_temporal_val.pkl',
]

# Logits
FEATURE_LOGIT_OUTPUT = 'data/intercepted_feature_logits'

# Energy Score
ENERGY_OUTPUT = 'data/energy_scores'

# Mahalanobis
FEATURE_OUTPUT = 'data/mahalanobis_distances'

# Data Plots
AUROC_PLOT_OUTPUT = 'plots/auroc'
FPR95_PLOT_OUTPUT = 'plots/fpr95'
RISK_COVERAGE_PLOT_OUTPUT = 'plots/risk_coverage'

BEVFORMER_TEST_BASE = 'core_models/BEVFormer/test/bevformer_base'

# Maps each (weather, severity) condition to its nuScenes-C ground truth PKL file
PKL_MAP = {
    ('Fog',  'easy'): 'data/sets/nuscenes-c/nuScenes-c/Fog/easy/nuscenes_infos_temporal_val.pkl',
    ('Fog',  'mid'):  'data/sets/nuscenes-c/nuScenes-c/Fog/mid/nuscenes_infos_temporal_val.pkl',
    ('Fog',  'hard'): 'data/sets/nuscenes-c/nuScenes-c/Fog/hard/nuscenes_infos_temporal_val.pkl',
    ('Snow', 'easy'): 'data/sets/nuscenes-c/nuScenes-c/Snow/easy/nuscenes_infos_temporal_val.pkl',
    ('Snow', 'mid'):  'data/sets/nuscenes-c/nuScenes-c/Snow/mid/nuscenes_infos_temporal_val.pkl',
    ('Snow', 'hard'): 'data/sets/nuscenes-c/nuScenes-c/Snow/hard/nuscenes_infos_temporal_val.pkl',
}

# Used for our own custom corruption generation (Not Needed anymore)
CORRUPTED_PKL = 'data/sets/nuscenes_corrupted/fog/nuscenes_infos_temporal_val.pkl'
OUTPUT_PKL = 'data/sets/nuscenes_combined/nuscenes_infos_temporal_MIXED_val.pkl'
