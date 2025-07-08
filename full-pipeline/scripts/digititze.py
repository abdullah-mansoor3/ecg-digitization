import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

from scipy.signal import medfilt

def remove_baseline_median(signal, fs, window_sec=0.6):
    """
    Removes baseline wander using sliding median filter.
    Args:
        signal (np.ndarray): ECG signal (1D)
        fs (int): Sampling frequency
        window_sec (float): Window size in seconds (default: 0.6s)
    Returns:
        np.ndarray: Detrended ECG signal
    """
    window_size = int(window_sec * fs)
    if window_size % 2 == 0:
        window_size += 1  # must be odd
    baseline = medfilt(signal, kernel_size=window_size)
    return signal - baseline

def correct_local_baseline(signal, fs=400, window_sec=2.0):
    """
    Correct ECG baseline drift by subtracting a local baseline in each window.
    Args:
        signal (np.ndarray): 1D ECG signal
        fs (int): Sampling frequency (default 400 Hz)
        window_sec (float): Window length in seconds (default 2s)
    Returns:
        np.ndarray: Baseline corrected signal
    """
    corrected = np.zeros_like(signal)
    window_size = int(window_sec * fs)
    n_windows = int(np.ceil(len(signal) / window_size))

    for i in range(n_windows):
        start = i * window_size
        end = min((i + 1) * window_size, len(signal))
        segment = signal[start:end]

        # Estimate baseline using lower percentile to avoid R peaks
        baseline = np.percentile(segment, 10)
        corrected[start:end] = segment - baseline

    return corrected

def normalize_around_flat_segments(signal, fs=400, window_sec=0.2, flat_thresh=0.05):
    """
    Normalize ECG signal by anchoring it to flat baseline segments (e.g., PR, ST, TP).
    
    Args:
        signal (np.ndarray): 1D ECG signal (in mV)
        fs (int): Sampling frequency
        window_sec (float): Window length in seconds to search for flat regions
        flat_thresh (float): Max std deviation in a window to consider it flat (in mV)
        
    Returns:
        np.ndarray: Baseline-shifted ECG signal
    """
    window_size = int(window_sec * fs)
    n = len(signal)
    flat_baselines = []

    for start in range(0, n - window_size, window_size // 2):
        window = signal[start:start + window_size]
        if np.std(window) < flat_thresh:
            flat_baselines.append(np.mean(window))

    if len(flat_baselines) == 0:
        print("Warning: No flat segments found. Returning original signal.")
        return signal.copy()

    estimated_baseline = np.median(flat_baselines)
    return signal - estimated_baseline




def binary_mask_to_waveform(mask, square_size):
    """
    Converts a binary ECG mask to a 1D waveform array in mV, sampled at 0.01s intervals.
    Baseline is found using the mode of all y indices (your logic).
    Removes trailing zero values after the wave ends.
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

    # --- Baseline logic ---
    if len(all_y_indices) == 0:
        raise ValueError("No wave pixels found in mask.")
    counts = np.bincount(all_y_indices, minlength=H)
    baseline_y = int(np.argmax(counts))

    # Amplitude in mV
    amplitude_mv = (baseline_y - y_means) * y_pixel_mV

    # Time axis (original resolution)
    time = np.arange(W) * x_pixel_sec

    # Interpolation to 0.0025s (40Hz)
    valid = ~np.isnan(amplitude_mv)
    interp_func = interp1d(time[valid], amplitude_mv[valid], kind='linear', bounds_error=False, fill_value="extrapolate")
    time_interp = np.arange(0, time[-1], 0.0025)
    signal_mv_interp = interp_func(time_interp)


    # Fix the baseline drift
    # signal_mv_interp = normalize_around_flat_segments(signal_mv_interp, fs=400)

    # signal_mv_interp = correct_local_baseline(signal_mv_interp, fs=400, window_sec=0.1)
    # signal_mv_interp = remove_baseline_median(signal_mv_interp, fs=400)

    # # Optionally clip large spikes
    # signal_mv_interp = np.clip(signal_mv_interp, -2, 2)


    # --- Trim trailing zeros ---
    non_zero_indices = np.where(np.abs(signal_mv_interp) > 1e-4)[0]
    if len(non_zero_indices) == 0:
        return np.array([]), np.array([]), baseline_y  # no signal found

    last_nonzero_idx = non_zero_indices[-1]
    signal_mv_interp = signal_mv_interp[:last_nonzero_idx + 1]
    time_interp = time_interp[:last_nonzero_idx + 1]

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
    skeleton = skeletonize(cleaned.astype(bool)).astype(np.uint8)

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
        # plt.plot(time_axis, waveform)
        # plt.xlabel("Time (s)")
        pass
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