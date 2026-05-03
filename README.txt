1. Install dependencies:
   pip install -r requirements.txt

2. Generate synthetic EEG dataset:
   python generate_data.py

3. Run full training pipeline (all plots + tables + model saving):
   python train.py

4. Launch web application:
   python app.py

5. Open browser at http://127.0.0.1:5000
   Upload any CSV file from the generated dataset (e.g., eeg_dataset/Healthy/subject_001.csv)
   
All expected outputs:
- Dataset distribution table (console or print)
- Signal comparison plot (original/filtered)
- Feature distribution graphs (saved as PNG)
- Sample-wise feature table (printed)
- SMOTE before/after table
- Model performance table (console)
- Confusion matrices (PNG)
- Accuracy comparison bar chart (highlight RF)
- Saved model 'best_eeg_model.pkl' and scaler
- Web app with prediction, waveform, peak highlighting, model summary
