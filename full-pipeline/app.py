import yaml
import os

with open('./configs/lead_segmentation.yaml', 'r') as f:
    lead_cfg = yaml.safe_load(f)
with open('./configs/digitize.yaml', 'r') as f:
    digitize_cfg = yaml.safe_load(f)

INPUT_IMAGE_DIR = lead_cfg['input_image_dir']
FINAL_OUTPUT_DIR = digitize_cfg['output_dir']

os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)