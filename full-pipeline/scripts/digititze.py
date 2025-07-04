import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

def binary_mask_to_waveform(mask, square_size):
    """
    Converts a binary ECG mask to a 1D waveform array in mV, sampled at 0.01s intervals.
    Baseline is found using the mode of all y indices (your logic).
    """
    H, W = mask.shape

    # Calibration
    x_pixel_sec = 0.2 / square_size   # seconds per pixel (horizontal)
    y_pixel_mV  = 0.5 / square_size   # mV per pixel (vertical)

    # For each column, get mean y of white pixels
    y_means = np.full(W, np.nan)
    all_y_indices = []
    for x in range(W):
        y_indices = np.where(mask[:, x] > 0)[0]
        if len(y_indices) > 0:
            y_means[x] = y_indices.mean()
            all_y_indices.extend(y_indices.tolist())

    # --- Baseline logic exactly as you wrote ---
    # Build a histogram of all y indices (vertical axis)
    if len(all_y_indices) == 0:
        raise ValueError("No wave pixels found in mask.")
    counts = np.bincount(all_y_indices, minlength=H)
    baseline_y = int(np.argmax(counts))

    # Amplitude: (y_baseline - y_mean) * y_pixel_mV
    amplitude_mv = (baseline_y - y_means) * y_pixel_mV
    # Above baseline: positive, below: negative

    # Time axis for original signal
    time = np.arange(W) * x_pixel_sec

    # Interpolate to 0.01s intervals
    time_interp = np.arange(0, time[-1], 0.0025)  # 0.0025s intervals (40Hz)
    valid = ~np.isnan(amplitude_mv)
    interp_func = interp1d(time[valid], amplitude_mv[valid], kind='linear', bounds_error=False, fill_value="extrapolate")
    signal_mv_interp = interp_func(time_interp)

    return signal_mv_interp, time_interp, baseline_y

def clean_and_skeletonize_wave(mask, min_area=100, gap_threshold=5):
    """
    Cleans and skeletonizes the binary ECG wave mask.
    Also connects vertically aligned segments and retains only the largest component.

    Args:
        mask (np.ndarray): Binary mask [H, W], wave = 1 or 255
        min_area (int): Minimum area to keep connected components.
        gap_threshold (int): Max vertical distance between centers to consider them aligned.

    Returns:
        np.ndarray: Skeletonized binary mask with only the main wave [H, W]
    """

    # Step 1: Remove small regions
    labeled = label(mask)
    cleaned = np.zeros_like(mask)
    regions = [r for r in regionprops(labeled) if r.area >= min_area]
    for r in regions:
        for y, x in r.coords:
            cleaned[y, x] = 1

    # Step 2: Attempt to connect vertically aligned components
    centers = [(int(r.centroid[0]), int(r.centroid[1])) for r in regions]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            y1, x1 = centers[i]
            y2, x2 = centers[j]
            if abs(x1 - x2) <= 1 and abs(y1 - y2) <= gap_threshold:
                # Draw line to connect components
                cv2.line(cleaned, (x1, y1), (x2, y2), 1, 1)

    # Step 3: Skeletonize
    # skeleton = skeletonize(cleaned.astype(bool)).astype(np.uint8)

    # Step 4: Keep only the largest connected component (assumed to be the main wave)
    labeled_skel = label(cleaned)
    props = regionprops(labeled_skel)

    if not props:
        return np.zeros_like(mask, dtype=np.uint8)

    largest = max(props, key=lambda x: x.area)
    final = np.zeros_like(mask, dtype=np.uint8)
    for y, x in largest.coords:
        final[y, x] = 255

    return final

def plot_waveform(waveform, time_axis=None, xlim=(0, 1000), ylim=(-1, 1), title="ECG Signal (in mV)"):
    """
    Plots the ECG waveform.
    Args:
        waveform (np.ndarray): 1D array of waveform values
        time_axis (np.ndarray or None): Optional time axis for x values
        xlim (tuple): x-axis limits
        ylim (tuple): y-axis limits
        title (str): Plot title
    """
    plt.figure(figsize=(10, 4))
    if time_axis is not None:
        plt.plot(time_axis, waveform)
        plt.xlabel("Time (s)")
    else:
        plt.plot(waveform)
        plt.xlabel("Sample")
    plt.title(title)
    plt.ylabel("Voltage (mV)")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid()
    plt.tight_layout()
    plt.show()

def process_ecg_mask(mask, square_size, output_dir=None, base_name=None, plot=False):
    """
    Processes a binary ECG mask and returns the waveform as a numpy array.
    Optionally saves the waveform array and plot if output_dir is provided.
    Args:
        mask (np.ndarray): Binary mask [H, W]
        square_size (float): Grid square size in pixels
        output_dir (str or None): Directory to save outputs (optional)
        base_name (str or None): Base name for saved files (optional)
        plot (bool): Whether to plot the waveform
    Returns:
        waveform (np.ndarray): 1D waveform array (interpolated)
    """
    # Clean and skeletonize
    mask = (mask > 0.5).astype(np.uint8)
    mask = np.squeeze(mask)
    mask = skeletonize(mask.astype(bool)).astype(np.uint8)

    waveform, time_axis, _ = binary_mask_to_waveform(mask, square_size=square_size)

    if plot:
        plot_waveform(waveform, time_axis=time_axis)

    if output_dir is not None and base_name is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{base_name}_wave.npy"), waveform)
        # Optionally save plot as image
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, waveform)
        plt.title("ECG Signal (in mV)")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.xlim((0, 1000))
        plt.ylim((-1, 1))
        plt.grid()
        plt.tight_layout()

    return waveform