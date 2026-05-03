import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from scipy import signal
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------- Load all saved artifacts ----------
model = joblib.load("best_eeg_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
with open("best_model_name.txt", "r") as f:
    best_model_name = f.read()
with open("best_accuracy.txt", "r") as f:
    best_accuracy = f.read()

# Pre‑computed metrics for all three models (replace with your actual values)
ALL_MODELS_METRICS = {
    "knn": {"accuracy": 0.85, "f1": 0.84, "roc_auc": 0.91, "cv_mean": 0.84, "cv_std": 0.02},
    "svm": {"accuracy": 0.88, "f1": 0.87, "roc_auc": 0.93, "cv_mean": 0.87, "cv_std": 0.03},
    "rf":  {"accuracy": 0.94, "f1": 0.93, "roc_auc": 0.97, "cv_mean": 0.93, "cv_std": 0.02}
}
CONFUSION_MATRIX = [[58, 2], [3, 57]]   # example: replace with your test confusion matrix

# Feature extraction (must match training)
from utils import load_eeg_file, preprocess_and_features, bandpass_filter  # in your utils.py

def compute_band_powers(eeg_signal, fs=128):
    """Return relative power (%) for delta, theta, alpha, beta."""
    freqs, psd = signal.welch(eeg_signal, fs=fs, nperseg=fs*2)
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    total = np.sum(psd)
    powers = {}
    for name, (low, high) in bands.items():
        idx = np.where((freqs >= low) & (freqs < high))[0]
        powers[name] = np.sum(psd[idx]) / total * 100
    return powers

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        # Load EEG signal (auto‑detect column)
        eeg_series = load_eeg_file(filepath)
        if len(eeg_series) == 0:
            raise ValueError("EEG signal is empty")
        
        # Preprocess & extract features (10 time‑domain features)
        filtered_sig, features = preprocess_and_features(eeg_series)
        
        # Scale & predict
        scaled = scaler.transform([features])
        pred_int = model.predict(scaled)[0]
        pred_proba = model.predict_proba(scaled)[0]
        predicted_class = label_encoder.inverse_transform([pred_int])[0]
        confidence = float(max(pred_proba)) * 100
        
        # Compute band powers from filtered signal
        band_powers = compute_band_powers(filtered_sig)
        
        # Return all data needed by frontend
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'band_powers': band_powers,
            'model_metrics': ALL_MODELS_METRICS,
            'confusion_matrix': CONFUSION_MATRIX,
            'model_name': best_model_name,
            'accuracy': best_accuracy
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)