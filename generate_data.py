import numpy as np
import pandas as pd
import os

FS = 128          # Hz
DURATION = 15     # seconds
N_SAMPLES = int(FS * DURATION)

CLASS_COUNTS = {
    "Healthy": 120,
    "Mild": 110,
    "Moderate": 95,
    "Severe": 85
}

def generate_eeg_signal(class_name, fs=FS, duration=DURATION):
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    np.random.seed(hash(class_name + str(np.random.randint(1e6))) % 2**32)
    
    # Clear separation: alpha/theta/delta amplitudes
    if class_name == "Healthy":
        # Strong alpha (8-12 Hz), low delta/theta
        eeg = 1.2 * np.sin(2*np.pi*10*t) + 0.9 * np.sin(2*np.pi*12*t) + 0.2 * np.sin(2*np.pi*6*t)
    elif class_name == "Mild":
        # Reduced alpha, increased theta
        eeg = 0.7 * np.sin(2*np.pi*10*t) + 1.1 * np.sin(2*np.pi*6*t) + 0.4 * np.sin(2*np.pi*3*t)
    elif class_name == "Moderate":
        # Theta/delta dominant, weak alpha
        eeg = 0.3 * np.sin(2*np.pi*10*t) + 1.3 * np.sin(2*np.pi*5*t) + 0.9 * np.sin(2*np.pi*2*t)
    else:  # Severe
        # Delta dominant, almost no alpha
        eeg = 0.1 * np.sin(2*np.pi*10*t) + 0.5 * np.sin(2*np.pi*4*t) + 1.6 * np.sin(2*np.pi*1.5*t)
    
    # Add realistic noise and subject variability
    eeg += 0.12 * np.random.randn(len(t))
    eeg *= np.random.uniform(0.7, 1.3)
    return eeg

def create_dataset():
    base_dir = "eeg_dataset"
    for class_name, count in CLASS_COUNTS.items():
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(count):
            eeg = generate_eeg_signal(class_name)
            pd.DataFrame({"EEG": eeg}).to_csv(os.path.join(class_dir, f"subject_{i+1:03d}.csv"), index=False)
    print("Dataset created with 4 distinct stages:")
    for cls, cnt in CLASS_COUNTS.items():
        print(f"  {cls}: {cnt} samples")

if __name__ == "__main__":
    create_dataset()