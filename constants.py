# NUSCENES contstants
OUTPUT_DIR = "data/sets/nuscenes/"
REGION = 'us'
FILES = {
    # Include the meta files at the top
    "v1.0-test_meta.tgz":"b0263f5c41b780a5a10ede2da99539eb",
    "v1.0-trainval_meta.tgz":"537d3954ec34e5bcb89a35d4f6fb0d4a",

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

# Check the kind of data you are downloading, and 
# uncomment the ones you need to create the corruption.
VERSIONS = [
    "v1.0-mini", 
    "v1.0-trainval", 
    "v1.0-test", 
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
