import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import label
from skimage.measure import regionprops

def process_ecg_mask(mask, square_size, output_dir=None, base_name=None, plot=False):
    """
    Converts a binary ECG wave mask to a 1D digitized ECG waveform based on square size calibration.

    Args:
        mask (np.ndarray): Binary mask [H, W] where wave pixels are > 0
        square_size (float): Size of one ECG square in pixels
        output_dir (str or None): Optional directory to save waveform
        base_name (str or None): Optional base name for saving
        plot (bool): Whether to show and/or save the waveform plot

    Returns:
        waveform (np.ndarray): 1D waveform (in mV)
    """

    # --- Step 1: Clean up mask ---
    mask = (mask > 0).astype(np.uint8)
    mask = np.squeeze(mask)

    # Optionally skeletonize (if wave is thick)
    skeleton = skeletonize(mask.astype(bool)).astype(np.uint8)

    # --- Step 2: Calculate calibration values ---
    pixels_per_sec = square_size / 0.2   # 5 squares = 1 sec → 1 square = 0.2s
    pixels_per_mV  = square_size / 0.5   # 1 square = 0.5 mV

    # So:
    seconds_per_pixel = 1.0 / pixels_per_sec
    mV_per_pixel      = 1.0 / pixels_per_mV

    H, W = skeleton.shape

    # --- Step 3: Find baseline ---
    horizontal_sum = skeleton.sum(axis=1)
    baseline_y = np.argmax(horizontal_sum)

    # --- Step 4: Extract waveform ---
    time_values = []
    amplitude_values = []

    for x in range(W):
        column = skeleton[:, x]
        white_pixels = np.where(column > 0)[0]

        if len(white_pixels) == 0:
            continue

        y = white_pixels[np.argmin(np.abs(white_pixels - baseline_y))]

        amplitude_mV = -(y - baseline_y) * mV_per_pixel
        time_sec = x * seconds_per_pixel

        time_values.append(time_sec)
        amplitude_values.append(amplitude_mV)

    waveform = np.array(amplitude_values)
    time_axis = np.array(time_values)

    # --- Step 5: Save plot and .npy if needed ---
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, waveform, color='blue')
        plt.title("ECG Signal (in mV)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (mV)")
        plt.grid(True)
        plt.tight_layout()
        if output_dir and base_name:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"{base_name}_waveform_plot.png"))
        plt.show()

    if output_dir and base_name:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{base_name}_wave.npy"), waveform)

    return waveform
