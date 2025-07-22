# Full Pipeline for Digitization of ECG Paper

This directory has the code (but not the models and data) to input a raw ecg paper, segment 12 leads (stored in .jpg), extract waves from the grid and digitize each wave (stored in .npy)

## Project Structure

- [configs](./configs/) : Config files (update accoridng to your need).
- [scripts](./scripts/) : Contains the following scripts (in order of execution in the pipeline) : 
  - [lead_segmentation](./scripts/lead_segmentation.py) : Crop the 12 leads using fine tuned YOLO model.
  - [grid_detection](./scripts/grid_detection.py) : Get the size of 1 big square in the grid. Used for calibration later on.
  - [extract_wave](./scripts/extract_wave.py) : Extract the binary mask of wave using the 97MB UNet model.
  - [extract_wave_s](./scripts/extract_wave_s.py) : Extract the binary mask of wave using the 7MB UNet model.
  - [digitize](./scripts/digitize.py) : Digitize the binary mask using the big square size (according to the 97 MB UNet model).
  - [digitize_s](./scripts/digitize_s.py) : Digitize the binary mask using the big square size (according to the 7 MB UNet model).
  - [create_ecg_paper](./scripts/create_ecg_paper.py) : Reconstruct the ECG paper and label the PQRST waves
- [main](./main.py) : Main entry point 

## How to run

### Create a Virtual Environment

```
conda create -y -p ./venv python=3.10
conda activate ./venv
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run

```
python3 main.py
```