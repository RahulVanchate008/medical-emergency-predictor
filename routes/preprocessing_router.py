from fastapi import APIRouter, File, UploadFile, HTTPException
import numpy as np
import pandas as pd
import wfdb
import os

router = APIRouter()

# Directory to store uploaded files
UPLOAD_DIR = "resources"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Endpoint to upload .dat, .hea, and .atr files and generate heart rate CSV
@router.post("/upload-ecg")
async def upload_ecg(
    dat_file: UploadFile = File(...), 
    hea_file: UploadFile = File(...), 
    atr_file: UploadFile = File(...)
):
    try:
        # Save the uploaded files to the UPLOAD_DIR
        dat_file_path = os.path.join(UPLOAD_DIR, dat_file.filename)
        hea_file_path = os.path.join(UPLOAD_DIR, hea_file.filename)
        atr_file_path = os.path.join(UPLOAD_DIR, atr_file.filename)

        with open(dat_file_path, "wb") as f:
            f.write(dat_file.file.read())
        with open(hea_file_path, "wb") as f:
            f.write(hea_file.file.read())
        with open(atr_file_path, "wb") as f:
            f.write(atr_file.file.read())
        
        # Extract record ID from file name (e.g., "100" from "100.dat")
        record_id = os.path.splitext(dat_file.filename)[0]

        # Read ECG data and annotations using wfdb
        record_path = os.path.join(UPLOAD_DIR, record_id)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')  # Use the provided .atr file

        # Extract signal and sampling frequency
        ecg_signal = record.p_signal[:, 0]  # Taking the first channel
        fs = record.fs

        # Use R-peaks from the annotation file
        r_peaks = annotation.sample  # Sample indices of R-peaks

        # Calculate RR intervals (time between consecutive R-peaks)
        rr_intervals = np.diff(r_peaks) / fs  # Convert to seconds

        # Calculate heart rate in beats per minute (BPM)
        heart_rate = 60 / rr_intervals  # BPM = 60 / RR interval in seconds

        # Save heart rates with timestamps
        timestamps = np.cumsum(rr_intervals)  # Cumulative time since start
        heart_rate_df = pd.DataFrame({
            'timestamp': timestamps,
            'heart_rate': heart_rate
        })

        # Save to CSV
        csv_path = os.path.join(UPLOAD_DIR, f"{record_id}_heart_rate.csv")
        heart_rate_df.to_csv(csv_path, index=False)

        return {"message": "Heart rate CSV generated successfully", "csv_path": csv_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
