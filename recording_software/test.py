import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import pickle

# --- CONFIGURATION ---
CONF = {
    "sample_rate": 1000, 
    "channels": 5,       
    "bandpass_low_hz": 20,
    "bandpass_high_hz": 450,
    "bandpass_order": 1,
    "mains_freq": 50.0,     
    "notch_order": 3,
}

class SignalProcessor:
    def __init__(self, fs: int):
        self.fs = float(fs)

        # High-pass (> 0.5 Hz)
        self.sos_hp = signal.butter(1, 0.5, btype="highpass", fs=self.fs, output="sos")

        # Band-pass (0.5 - 8 Hz)
        self.sos_bp = signal.butter(
            CONF["bandpass_order"],
            [CONF["bandpass_low_hz"], CONF["bandpass_high_hz"]],
            btype="bandpass",
            fs=self.fs,
            output="sos",
        )

        # Notch Filter (50 Hz + Harmonics)
        self.sos_notches = []
        f0 = float(CONF["mains_freq"])
        nyq = self.fs / 2.0
        k = 1
        while True:
            fk = k * f0
            if fk >= nyq: break
            sos = signal.butter(CONF["notch_order"], [fk-1, fk+1], btype='bandstop', fs=self.fs, output='sos')
            self.sos_notches.append(sos)
            k += 1

    def process(self, x: np.ndarray) -> np.ndarray:
        y = x.astype(np.float64, copy=True)

        # 1. Normalize initial values
        y -= y[0, :]

        # 2. High-pass
        y = signal.sosfiltfilt(self.sos_hp, y, axis=0)

        # 3. Center
        y -= np.mean(y, axis=0, keepdims=True)

        # 4. Notch (50 Hz)
        for sos in self.sos_notches:
            y = signal.sosfiltfilt(sos, y, axis=0)

        # 5. Band-pass (0.5 - 8 Hz)
        y = signal.sosfiltfilt(self.sos_bp, y, axis=0)

        return y.astype(np.float32)

def compute_psd(data, fs):
    freqs, psd = signal.welch(data, fs=fs, nperseg=250, axis=0, scaling='density')
    return freqs, psd

def inspect_pkl(filename: str):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        print("Generating dummy data (250Hz, 50Hz mains noise)...")
        fs = CONF["sample_rate"]
        duration = 3.6
        t = np.linspace(0, duration, int(fs*duration))
        # Simulate signal with 50Hz noise instead of 60Hz
        raw_data = np.sin(2 * np.pi * 2 * t[:, None] + np.random.rand(1, CONF["channels"])) * 50
        raw_data += np.sin(2 * np.pi * 50 * t[:, None]) * 20 
        raw_data += np.random.normal(0, 5, raw_data.shape)
    else:
        print(f"Loading {filename}...")
        with open(filename, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            raw_data = np.asarray(data["raw_uV"])
            fs = int(data.get("fs", CONF["sample_rate"]))
        else:
            raw_data = np.asarray(data)
            fs = CONF["sample_rate"]

    print(f"Input Shape: {raw_data.shape} | Fs: {fs} Hz")

    processor = SignalProcessor(fs)
    proc_data = processor.process(raw_data)
    t = np.arange(proc_data.shape[0]) / fs

    # --- PLOTTING ---
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

    # 1. Time Domain
    ax1 = fig.add_subplot(gs[0, :]) 
    for i in range(min(proc_data.shape[1], CONF["channels"])):
        ax1.plot(t, proc_data[:, i], label=f"Ch{i+1}", alpha=0.8, linewidth=1.5)
    
    ax1.set_title(f"Processed Neuromuscular Signals (0.5 - 8.0 Hz)")
    ax1.set_ylabel("Amplitude (µV)")
    ax1.set_xlim(0, t[-1])
    ax1.legend(loc='upper right', ncol=5, fontsize='small')
    ax1.grid(True, alpha=0.3)

    # 2. Zoomed View
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(min(proc_data.shape[1], CONF["channels"])):
        ax2.plot(t, proc_data[:, i], label=f"Ch{i+1}", alpha=0.8)
    
    mid_point = t[-1] / 2
    ax2.set_xlim(mid_point - 0.5, mid_point + 0.5) 
    ax2.set_title("Zoomed View (1s)")
    ax2.set_ylabel("Amplitude (µV)")
    ax2.grid(True, alpha=0.3)

    # 3. Frequency Domain
    ax3 = fig.add_subplot(gs[1, 1])
    f_proc, p_proc = compute_psd(proc_data, fs)
    
    for i in range(min(proc_data.shape[1], CONF["channels"])):
        ax3.semilogy(f_proc, p_proc[:, i], label=f"Ch{i+1}")

    ax3.set_title("PSD of Processed Signals")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_xlim(0, 70) 
    ax3.grid(True, which="both", alpha=0.3)
    
    # Markers for 8Hz cutoff and 50Hz notch
    ax3.axvline(8.0, color='r', linestyle='--', alpha=0.5, label="8Hz Cutoff")
    ax3.axvline(50.0, color='k', linestyle=':', alpha=0.5, label="50Hz Notch")
    ax3.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    FILENAME = "Harvard_EMG_Dataset/Subject_01/Session_20251225/s01_sent332_202209.pkl"
    inspect_pkl(FILENAME)