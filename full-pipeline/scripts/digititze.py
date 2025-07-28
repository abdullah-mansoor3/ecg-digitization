import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.interpolate import interp1d
import pandas as pd
from pandas import Series
from scipy.signal import butter, filtfilt
from scipy.signal import medfilt

def trim_signal_edges(signal, threshold=1e-4):
    signal = np.array(signal)
    print(f"[DEBUG] Trimming signal with {len(signal)} points, threshold = {threshold}")

    valid_mask = ~np.isnan(signal) & (np.abs(signal) > threshold)
    if not np.any(valid_mask):
        print("[WARN] All signal values are below threshold or NaN.")
        return np.array([])

    start = np.argmax(valid_mask)
    end = len(signal) - np.argmax(valid_mask[::-1])
    print(f"[DEBUG] Trimmed signal from index {start} to {end} ({end - start} samples)")
    return signal[start:end]


def highpass_filter(signal, fs, cutoff=0.5, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered

def lowpass_filter(signal, fs, cutoff=40.0, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered

def trim_low_energy_and_spikes(signal, time=None, threshold_ratio=0.05, window_size=50, spike_threshold=5.0):
    print(f"[DEBUG] Trimming low energy and spikes: len(signal) = {len(signal)}")
    energy = np.convolve(signal**2, np.ones(window_size), mode='same')
    max_energy = np.max(energy)
    threshold = threshold_ratio * max_energy
    print(f"[DEBUG] Energy threshold: {threshold:.6f}, max energy: {max_energy:.6f}")

    nonzero_indices = np.where(energy > threshold)[0]
    if len(nonzero_indices) == 0:
        print("[WARN] No signal regions above energy threshold")
        return signal, time

    start_idx = nonzero_indices[0]
    end_idx = nonzero_indices[-1]
    print(f"[DEBUG] Trimming based on energy from {start_idx} to {end_idx}")

    signal_trimmed = signal[start_idx:end_idx + 1]
    time_trimmed = time[start_idx:end_idx + 1] if time is not None else None

    diff = np.diff(signal_trimmed)
    spike_indices = np.where(np.abs(diff) > spike_threshold)[0]
    print(f"[DEBUG] Found {len(spike_indices)} spike(s) over threshold of {spike_threshold}")

    for idx in spike_indices:
        if idx < len(signal_trimmed) * 0.05:
            print(f"[DEBUG] Removing spike near start at index {idx}")
            signal_trimmed = signal_trimmed[idx+1:]
            if time_trimmed is not None:
                time_trimmed = time_trimmed[idx+1:]
        elif idx > len(signal_trimmed) * 0.95:
            print(f"[DEBUG] Removing spike near end at index {idx}")
            signal_trimmed = signal_trimmed[:idx]
            if time_trimmed is not None:
                time_trimmed = time_trimmed[:idx]

    return signal_trimmed, time_trimmed


def process_ecg_mask(mask, square_size, output_dir=None, base_name=None, plot=False):
    print(f"\n[INFO] Processing ECG mask for {base_name if base_name else 'Unnamed'}")

    # Step 1
    mask = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(mask.astype(bool)).astype(np.uint8)
    H, W = skeleton.shape
    print(f"[DEBUG] Skeletonized mask shape: {H} x {W}")

    # Step 2
    x_pixel_sec = 0.2 / square_size
    y_pixel_mV  = 0.5 / square_size
    print(f"[DEBUG] Calibration: {x_pixel_sec:.4f}s/pixel (X), {y_pixel_mV:.4f}mV/pixel (Y)")

    # Step 3
    y_means = np.full(W, np.nan)
    all_y_indices = []
    for x in range(W):
        y_indices = np.where(skeleton[:, x] > 0)[0]
        if len(y_indices) > 0:
            y_means[x] = y_indices.mean()
            all_y_indices.extend(y_indices.tolist())

    if len(all_y_indices) == 0:
        print(f"[WARN] No wave pixels found in skeleton for {base_name}")
        return np.array([])

    # Step 4
    counts = np.bincount(all_y_indices, minlength=H)
    baseline_y = int(np.argmax(counts))
    print(f"[DEBUG] Baseline Y position: {baseline_y}")

    # Step 5
    amplitude_mv = trim_signal_edges((baseline_y - y_means) * y_pixel_mV)
    if len(amplitude_mv) == 0:
        print(f"[WARN] Amplitude signal is empty after trimming in {base_name}")
        return np.array([])
    amplitude_mv = pd.Series(amplitude_mv).interpolate(method='linear', limit_direction='both').to_numpy()
    time = np.arange(len(amplitude_mv)) * x_pixel_sec

    # Step 6
    valid = ~np.isnan(amplitude_mv)
    if np.sum(valid) < 2:
        print(f"[ERROR] Not enough valid points for interpolation in {base_name}")
        return np.array([])

    interp_func = interp1d(time[valid], amplitude_mv[valid], kind='linear', bounds_error=False)
    time_interp = np.arange(0, time[-1], 0.0025)
    signal_mv_interp = interp_func(time_interp)
    print(f"[DEBUG] Interpolated to 400Hz, new length: {len(signal_mv_interp)}")

    # Step 7
    non_zero_indices = np.where(np.abs(signal_mv_interp) > 1e-4)[0]
    if len(non_zero_indices) == 0:
        print("[WARN] Interpolated signal is all near-zero")
        return np.array([])

    last_nonzero_idx = non_zero_indices[-1]
    signal_mv_interp = signal_mv_interp[:last_nonzero_idx + 1]
    time_interp = time_interp[:last_nonzero_idx + 1]

    # Step 8
    signal_filtered = highpass_filter(signal_mv_interp, fs=400)
    signal_filtered = lowpass_filter(signal_filtered, fs=400)
    signal_clean = medfilt(signal_filtered, kernel_size=3)
    signal_trimmed, time_trimmed = trim_low_energy_and_spikes(signal_clean, time_interp)
    print(f"[INFO] {base_name} processed signal length: {len(signal_trimmed)} samples")



    # --- Step 8: Plot if needed ---
    if plot:
        plt.figure(figsize=(20,5))
        plt.plot(time_trimmed, signal_trimmed, color='black')
        plt.title("Final Noise Removed ECG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (mV)")
        plt.grid(True)
        plt.tight_layout()
        if output_dir and base_name:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"{base_name}_waveform_plot.png"))
        plt.show()

    # --- Step 9: Save waveform if needed ---
    if output_dir and base_name:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{base_name}_wave.npy"), signal_trimmed)

    return signal_trimmed
