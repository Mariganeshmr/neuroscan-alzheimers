import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

FS = 128  # sampling frequency

def bandpass_filter(eeg_signal, lowcut=0.5, highcut=50, fs=FS, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, eeg_signal)

def extract_time_features(eeg_signal):
    """10 time-domain features"""
    mean_val = np.mean(eeg_signal)
    std_val = np.std(eeg_signal)
    var_val = np.var(eeg_signal)
    power_val = np.mean(eeg_signal ** 2)
    energy_val = np.sum(eeg_signal ** 2)
    rms_val = np.sqrt(np.mean(eeg_signal ** 2))
    peak_to_peak = np.ptp(eeg_signal)
    zero_crossings = ((np.diff(np.sign(eeg_signal)) != 0).sum()) / len(eeg_signal)
    
    # Hjorth parameters
    if var_val != 0:
        hjorth_mobility = np.sqrt(np.var(np.diff(eeg_signal)) / var_val)
    else:
        hjorth_mobility = 0
    if hjorth_mobility != 0 and len(eeg_signal) > 2:
        hjorth_complexity = (np.var(np.diff(np.diff(eeg_signal))) / np.var(np.diff(eeg_signal))) / hjorth_mobility
    else:
        hjorth_complexity = 0
    
    return [mean_val, std_val, var_val, power_val, energy_val,
            rms_val, peak_to_peak, zero_crossings,
            hjorth_mobility, hjorth_complexity]

def preprocess_and_features(eeg_series):
    sig = eeg_series.values if isinstance(eeg_series, pd.Series) else np.array(eeg_series)
    filtered = bandpass_filter(sig)
    features = extract_time_features(filtered)
    return filtered, features

def load_eeg_file(csv_path):
    """Auto-detect EEG column"""
    df = pd.read_csv(csv_path)
    # Common column names for EEG
    for col in ['EEG', 'eeg', 'signal', 'value', 'Fp1', 'C3', 'P3', 'Fp2']:
        if col in df.columns:
            print(f"Using column '{col}' as EEG signal")
            return df[col]
    # Take first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"Using first numeric column '{numeric_cols[0]}' as EEG")
        return df[numeric_cols[0]]
    raise ValueError("No numeric EEG column found")

def detect_peaks_and_abnormal(signal, fs=FS):
    peaks, _ = find_peaks(signal, distance=fs*0.3, height=np.std(signal)*0.5)
    threshold = np.mean(signal) + 2 * np.std(signal)
    abnormal_mask = np.abs(signal) > threshold
    return peaks, abnormal_mask

def plot_eeg_with_highlights(signal, peaks, abnormal_mask, title="EEG Signal"):
    fig, ax = plt.subplots(figsize=(12, 4))
    time = np.arange(len(signal)) / FS
    ax.plot(time, signal, color='#1f77b4', label='EEG')
    ax.plot(time[peaks], signal[peaks], 'ro', markersize=4, label='Peaks')
    if np.any(abnormal_mask):
        ax.fill_between(time, -1, 1, where=abnormal_mask, alpha=0.3, color='red', label='Abnormal')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()
    return fig