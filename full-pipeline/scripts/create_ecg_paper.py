import os
import numpy as np
import matplotlib.pyplot as plt
from scripts.detect_pqrs import detect_pqrs  # Import the peak detection function

def create_ecg_paper(leads, labels, output_path, grid_size=0.5, paper_sec=10, fs=400):
    """
    Reconstructs a 6x2 ECG paper from lead waveforms and saves as an image, marking P, Q, R, S, T peaks.
    
    Args:
        leads (list of np.ndarray): List of 1D numpy arrays (waveforms in mV).
        labels (list of str): List of lead names (e.g., ['I', 'II', 'III', ...]).
        output_path (str): Path to save the output image.
        grid_size (float): Grid size in mV and seconds (default 0.5 mV, 0.2s).
        paper_sec (int): Duration of paper in seconds (default 10s).
        fs (int): Sampling frequency of waveforms (Hz).
    """
    assert len(leads) == len(labels), "Leads and labels must be same length"
    assert len(leads) <= 12, "Supports up to 12 leads"

    # Pad/cut all leads to the same length (paper_sec)
    n_samples = int(paper_sec * fs)
    leads_proc = []
    for w in leads:
        if len(w) < n_samples:
            pad = np.full(n_samples - len(w), np.nan)
            w = np.concatenate([w, pad])
        else:
            w = w[:n_samples]
        leads_proc.append(w)
    leads_proc = np.array(leads_proc)

    # 6x2 grid
    n_rows, n_cols = 6, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    time = np.arange(n_samples) / fs

    # ECG grid background
    def draw_ecg_grid(ax, sec, mv, fs, grid_size=0.5):
        # Major grid: 0.2s, 0.5mV; Minor: 0.04s, 0.1mV
        xlim = (0, sec)
        ylim = (-2, 2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # Minor grid
        for x in np.arange(0, sec, 0.04):
            ax.axvline(x, color='#f8cccc', lw=0.5, zorder=0)
        for y in np.arange(-2, 2.1, 0.1):
            ax.axhline(y, color='#f8cccc', lw=0.5, zorder=0)
        # Major grid
        for x in np.arange(0, sec, 0.2):
            ax.axvline(x, color='#e88', lw=1, zorder=0)
        for y in np.arange(-2, 2.1, 0.5):
            ax.axhline(y, color='#e88', lw=1, zorder=0)
        ax.set_xticks(np.arange(0, sec+0.1, 0.2))
        ax.set_yticks(np.arange(-2, 2.1, 0.5))
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Colors for each wave
    wave_colors = {"P": "green", "Q": "purple", "R": "red", "S": "orange", "T": "blue"}

    # Plot each lead
    for idx, (lead, label) in enumerate(zip(leads_proc, labels)):
        row = idx % n_rows
        col = idx // n_rows
        ax = axes[row, col]
        draw_ecg_grid(ax, paper_sec, 2, fs)
        # Detect peaks and get cleaned signal
        peaks, cleaned = detect_pqrs(lead, sampling_rate=fs)
        ax.plot(time, cleaned, color='black', lw=1.5)
        ax.text(0.01, 0.85, label, transform=ax.transAxes, fontsize=16, fontweight='bold', color='darkred')
        # Plot detected peaks
        for wave, color in wave_colors.items():
            if wave in peaks and len(peaks[wave]) > 0:
                ax.plot(time[peaks[wave]], cleaned[peaks[wave]], 'o', label=wave, color=color, markersize=4)
        # Optionally, show legend for the first subplot only
        if row == 0 and col == 0:
            ax.legend(loc='upper right', fontsize=10)

    # Hide unused axes if <12 leads
    for idx in range(len(leads), n_rows * n_cols):
        row = idx % n_rows
        col = idx // n_rows
        axes[row, col].axis('off')

    fig.suptitle("Reconstructed 6x2 ECG Paper with PQRST Peaks", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)