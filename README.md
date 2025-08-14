# ECG Digitization Project

## Overview

This project provides a **full pipeline** for digitizing 12-lead ECG paper recordings. The pipeline takes scanned ECG images, segments the 12 leads, extracts the waveform from each lead, calibrates the signal using grid detection, and outputs digitized signals for further analysis (including PQRST detection and myocardial infarction prediction).

## Project Structure

```
full-pipeline/
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── test.ipynb
├── configs/
│   ├── lead_segmentation.yaml
│   ├── wave_extraction.yaml
│   ├── grid_detection.yaml
│   └── digitize.yaml
├── data/
│   ├── inputs/      # Raw ECG images (input)
│   └── outputs/     # Cropped leads, binary masks, digitized signals (output)
├── scripts/
│   ├── lead_segmentation.py
│   ├── grid_detection.py
│   ├── extract_wave.py
│   ├── digititze.py
│   ├── create_ecg_paper.py
│   ├── analyze_waves.py
│   └── ecg_dataset.py
```

- **configs/**: YAML configuration files for each pipeline stage.
- **data/inputs/**: Place your raw ECG paper images here.
- **data/outputs/**: All intermediate and final outputs are saved here.
- **scripts/**: Modular scripts for each step of the pipeline.
- **main.py**: Main entry point for running the full pipeline.

## Pipeline Methodology

### 1. Lead Segmentation (YOLO)
- Uses a fine-tuned YOLO model to detect and crop the 12 leads from the input ECG image.
- Each lead is saved as a separate image in `data/outputs/`.
- Configuration: `./full-pipeline/configs/lead_segmentation.yaml`

### 2. Grid Detection & Calibration
- Detects the grid on the ECG paper using morphological operations.
- Estimates the pixel size of one large grid square for calibration (0.2s horizontally, 0.5mV vertically).
- Configuration: `./full-pipeline/configs/grid_detection.yaml`

### 3. Wave Extraction (U-Net)
- Applies a trained U-Net model to each cropped lead image to extract the binary mask of the ECG waveform.
- Removes background and isolates the signal.
- Configuration: `./full-pipeline/configs/wave_extraction.yaml`

### 4. Digitization
- Converts the binary mask to a 1D signal array using the grid calibration.
- Interpolates the signal to a uniform time axis.
- Saves the digitized waveform as `.npy` files in `./full-pipeline/data/outputs/`.
- Configuration: `./full-pipeline/configs/digitize.yaml`

### 5. ECG Paper Reconstruction & Analysis (Optional)
- Reconstructs the full ECG paper from digitized leads.
- Optionally analyzes the signals for PQRST peaks and other features.

## How to Run

In the full-pipeline folder

1. **Set up environment**
   ```sh
   conda create -y -p ./venv python=3.10
   conda activate ./venv
   pip install -r requirements.txt
   ```

2. **Place your ECG images**
   - Put raw ECG images in `./full-pipeline/data/inputs/`.

3. **Configure pipeline**
   - Edit YAML files in `./full-pipeline/configs/` as needed (paths, model weights, parameters).

4. **Run the pipeline**
   ```sh
   python3 main.py
   ```

5. **Find results**
   - Cropped leads, binary masks, digitized signals, and reconstructed ECG paper will be in `./full-pipeline/data/outputs/`.

## Methodologies Used

- **Object Detection (YOLO):** For robust segmentation of ECG leads.
- **Morphological Image Processing:** For grid detection and calibration.
- **Semantic Segmentation (U-Net):** For extracting the ECG waveform from noisy backgrounds.
- **Signal Processing:** For converting binary masks to calibrated 1D signals.
- **Configurable Pipeline:** All parameters and paths are set via YAML files for reproducibility.

## Notes

- The pipeline currently uses a PyTorch U-Net model. For deployment or speed, you may switch to ONNX or TFLite models.
- All intermediate outputs are saved for transparency and debugging.
- The pipeline is modular—each step can be run and debugged independently.

**For more details, see the scripts and notebooks in the `./full-pipeline/scripts/`, `lead-segmentation/`, `wave-binary-mask/` and `notebooks/` directories.**
