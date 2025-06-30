# Step 1: Preprocessing

**Techniques**: CLAHE, denoising, grayscale conversion

**Purpose**: Enhance ECG trace visibility and reduce noise before detection or segmentation

**File Path** : [Augmentation script](./lead-segmentation/scripts/augment-dataset.py)

# Step 2: Lead and Region Detection
**Model**: YOLOv8 (object detection)

**Input**: Full ECG paper image

**Output**: Cropped images of each lead, metadata, calibration segment

**Purpose**: Isolate individual ECG leads for separate processing and clean signal extraction

# Step 3: Perspective Correction
**Techniques**: Hough Transform (for big box), corner detection, cv2.getPerspectiveTransform

**Purpose**: Remove skew/tilt so that pixel distances are uniform across the image for accurate time-voltage calibration

# Step 4: Trace Segmentation
**Model**: U-Net (segmentation)

**Input**: Single-lead cropped image

**Output**: Binary mask (1 = waveform, 0 = background/grid)

**Purpose**: Separate ECG signal from dotted grid and noise for clean digitization

# Step 5: Grid Analysis and Calibration
**Techniques**: Blob detection (for dots), clustering, spacing analysis

**Purpose**: Calculate pixel-to-time (s/pixel) and pixel-to-voltage (mV/pixel) scaling using spacing between small grid dots

# Step 6: Digitization (1D Signal Extraction)
**Techniques**: Column-wise vertical scanning on binary mask

**Purpose**: Convert 2D trace into a time-series signal by mapping each column’s trace position to voltage using grid calibration

**Optional**: GAN-Based Digitization
**Model**: Pix2Pix (conditional GAN)

**Purpose**: Learn direct image-to-waveform mapping as an alternative to scanning + calibration for higher robustness

# Output
Clean 1D digital ECG signals (CSV/NumPy), stored for further analysis, plotting, or classification

