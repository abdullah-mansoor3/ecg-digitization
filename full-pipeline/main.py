import time
start_time = time.time()

import os
import cv2
import yaml
import torch
import numpy as np
from PIL import Image

from scripts.grid_detection import get_grid_square_size
from scripts.extract_wave_tflite import WaveExtractor
from scripts.digititze import process_ecg_mask
from scripts.create_ecg_paper import create_ecg_paper
from scripts.lead_segmentation_tflite import init_model as init_lead_model, inference_and_label_and_crop
from scripts.analyze_waves import analyze_waves
# --- Load configs ---
with open('./configs/lead_segmentation.yaml', 'r') as f:
    lead_cfg = yaml.safe_load(f)
with open('./configs/wave_extraction.yaml', 'r') as f:
    wave_cfg = yaml.safe_load(f)
with open('./configs/grid_detection.yaml', 'r') as f:
    grid_cfg = yaml.safe_load(f)
with open('./configs/digitize.yaml', 'r') as f:
    digitize_cfg = yaml.safe_load(f)

# --- Paths ---
INPUT_IMAGE_DIR = lead_cfg['input_image_dir']
CROPPED_SAVE_DIR = lead_cfg['output_dir']
GRID_KERNEL = grid_cfg.get('closing_kernel', 10)
GRID_LENGTH_FRAC = grid_cfg.get('length_frac', 0.05)
WAVE_WEIGHTS_PATH = wave_cfg['weights_path']
WAVE_DEVICE = wave_cfg.get('device', 'cpu')
FINAL_OUTPUT_DIR = digitize_cfg['output_dir']
YOLO_WEIGHTS_PATH = lead_cfg['model_path']
os.makedirs(CROPPED_SAVE_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

# --- 1. Lead Segmentation ---
print("Running lead segmentation...")
lead_model = init_lead_model(lead_cfg['model_path'])
image_files = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
base_name = [file.replace('.jpg', '').replace('.jpeg', '').replace('.png', '') for file in image_files]

all_cropped_leads = []
for img_file, base_name in zip(image_files, base_name):

    img_path = os.path.join(INPUT_IMAGE_DIR, img_file)
    img = cv2.imread(img_path)
    cropped_leads = inference_and_label_and_crop(
        lead_model, img, conf_threshold=lead_cfg['conf_threshold']
    )

    for crop_img, label in cropped_leads:

        original_size = crop_img.shape[:2]  # (height, width)

        #save to file(optional)
        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f'{base_name}_{label}_crop.jpg'), crop_img)

        # Save original size for resizing masks later
        all_cropped_leads.append((crop_img, label, original_size))



# --- 2. Grid Detection & Square Size Estimation ---
print("Estimating grid square sizes...")
lead_to_square_size = {}
for crop_img, label, original_size in all_cropped_leads:
    
    if crop_img is None:
        print(f"No image found for {label}")
        continue
    square_size = get_grid_square_size(crop_img, closing_kernel=GRID_KERNEL, length_frac=GRID_LENGTH_FRAC)
    print(f"Estimated square size for {label}: {square_size} pixels")
    lead_to_square_size[label] = square_size

# --- 3. Wave Extraction (Binary Mask) ---
print("Extracting binary wave masks...")
wave_extractor = WaveExtractor(WAVE_WEIGHTS_PATH, device=WAVE_DEVICE)
lead_to_wave_mask = {}

for crop_img, label, original_size in all_cropped_leads:
    print(f"Processing: {label}")
    # print(f"Original size raw: {original_size} (type: {type(original_size)})")

    try:
        h, w = int(original_size[1]), int(original_size[0])
    except Exception as e:
        print(f"  ❌ ERROR parsing original_size: {original_size}")
        raise e

    binary_mask = wave_extractor.extract_wave(Image.fromarray(crop_img))
    binary_mask_np = np.array(binary_mask)

    binary_mask_resized = cv2.resize(binary_mask_np, (h, w), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f'{base_name}_{label}_binarymask.jpg'), binary_mask_resized)

    lead_to_wave_mask[label] = binary_mask_resized
    # print(f"✅ Resized binary mask for {base_name}_{label}: {binary_mask_resized.shape}")

    # wave_extractor.save_wave(binary_mask,os.path.join(FINAL_OUTPUT_DIR, f"{base_name}_{label}_binary_wave_plot.png"))
    
# --- 4. Digitize: Convert Mask to Waveform ---
print("Digitizing waveforms...")
lead_waveforms = []
lead_labels = []
for crop_path, label, original_size in all_cropped_leads:
    print(f'Processing {label}')
    # original_size = (height, width)
    h_orig, w_orig = original_size
    # print(f"[DEBUG] {base_name}_{label} crop_img shape: height={h_orig}, width={w_orig}")
    binary_mask = lead_to_wave_mask[label]
    # print(f"[DEBUG]   raw mask shape: {binary_mask.shape}")

    square_size = lead_to_square_size[label]
    # print(f"[DEBUG]   square_size: {square_size}")

    # *** FIX: pass (w_orig, h_orig) to resize ***
    binary_mask_resized = cv2.resize(
        binary_mask,
        (w_orig, h_orig),                # width, height
        interpolation=cv2.INTER_NEAREST
    )
    # print(f"[DEBUG]   resized mask shape: {binary_mask_resized.shape}")

    waveform = process_ecg_mask(
        binary_mask_resized,
        square_size,
        # plot_path = os.path.join(FINAL_OUTPUT_DIR, f'{base_name}_{label}_plot.jpg'), #optional
        array_path = os.path.join(FINAL_OUTPUT_DIR, f'{base_name}_{label}.npy') #optional
    )
    # print(f"[DEBUG]   digitized waveform length: {len(waveform)} samples\n")
    lead_waveforms.append(waveform)
    lead_labels.append(label)

    


# --- 5. Give Final Analysis ---
print('Analyzing Waves...')
print(f'Final Analaysis: {analyze_waves(lead_waveforms, lead_labels)}')




# --- Optional Create ECG Paper ---
# print("Creating ECG paper with all leads...")
# ecg_paper_path = os.path.join(FINAL_OUTPUT_DIR, "reconstructed_ecg_paper.png")
# create_ecg_paper(lead_waveforms, lead_labels, ecg_paper_path)
# print("ECG paper saved to:", ecg_paper_path)


print("Pipeline complete. All outputs saved to:", FINAL_OUTPUT_DIR)

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")