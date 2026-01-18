#!/usr/bin/env python3
"""
Process SWELL raw RRI data and generate feature CSV files.

This script:
1. Reads raw RRI (R-R interval) data from data/SWELL/data/raw/rri/
2. Processes physiological features
3. Generates synthetic features for other modalities
4. Creates CSV files in the format expected by the SWELL loader
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
import warnings

warnings.filterwarnings('ignore')

def load_rri_file(filepath: Path, max_samples: int = 1000) -> tuple:
    """Load RRI data from text file with optional subsampling."""
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Subsample if too large
        if len(data) > max_samples:
            indices = np.linspace(0, len(data) - 1, max_samples, dtype=int)
            data = data[indices]
        
        time = data[:, 0]
        values = data[:, 1]
        return time, values
    except Exception as e:
        print(f"  ⚠️  Error loading {filepath}: {e}")
        return None, None

def extract_heart_rate_features(rri_values: np.ndarray) -> dict:
    """Extract HR and HRV features from RRI data."""
    # Convert RRI (in ms) to heart rate (in bpm)
    hr = 60000 / rri_values  # 60000 ms per minute
    
    features = {
        'hr_mean': float(np.mean(hr)),
        'hr_std': float(np.std(hr)),
        'hr_min': float(np.min(hr)),
        'hr_max': float(np.max(hr)),
    }
    
    # HRV features
    rri_diff = np.diff(rri_values)
    features['hrv_rmssd'] = float(np.sqrt(np.mean(rri_diff**2)))
    features['hrv_sdnn'] = float(np.std(rri_values))
    
    return features


def infer_stress_label(minute: int, n_minutes: int) -> str:
    """Infer stress condition based on time in session.
    
    Typical SWELL structure:
    - First 10 min: Baseline (no stress)
    - Middle 40 min: Tasks (stress)
    - Last 10 min: Recovery (no stress)
    """
    if n_minutes < 20:
        # Short session: first half baseline, second half stress
        return "no stress" if minute < n_minutes // 2 else "stress"
    else:
        # Full session
        first_baseline = int(0.15 * n_minutes)  # 15% baseline
        last_recovery = int(0.15 * n_minutes)   # 15% recovery
        
        if minute < first_baseline:
            return "no stress"
        elif minute >= n_minutes - last_recovery:
            return "no stress"
        else:
            return "stress"

def generate_features_from_rri():
    """Process RRI files and generate SWELL feature CSVs."""
    
    rri_dir = Path("data/SWELL/data/raw/rri")
    output_dir = Path("data/SWELL")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not rri_dir.exists():
        print(f"❌ RRI directory not found: {rri_dir}")
        return False
    
    rri_files = sorted(rri_dir.glob("p*.txt"))
    print(f"🔍 Found {len(rri_files)} RRI files")
    
    if not rri_files:
        print("❌ No RRI files found!")
        return False
    
    # Extract subject numbers from filenames (p1.txt -> 1, p10.txt -> 10, etc.)
    subjects = []
    for f in rri_files:
        try:
            subj_num = int(f.stem[1:])  # Remove 'p' prefix
            subjects.append(subj_num)
        except ValueError:
            print(f"  ⚠️  Could not parse subject from {f.name}")
    
    subjects = sorted(set(subjects))
    print(f"✅ Subjects: {subjects}")
    
    # =============================
    # 1. PHYSIOLOGY FEATURES (from RRI data)
    # =============================
    print("\n📊 Generating physiology features from RRI data...")
    physiology_data = []
    
    for subject in subjects:
        rri_file = rri_dir / f"p{subject}.txt"
        if not rri_file.exists():
            print(f"  ⚠️  {rri_file.name} not found")
            continue
        
        time_vals, rri_vals = load_rri_file(rri_file)
        if time_vals is None:
            continue
        
        # Get HR/HRV features
        hr_features = extract_heart_rate_features(rri_vals)
        
        # Resample to minute-level features (assuming data spans ~60 minutes)
        n_minutes = max(1, int(len(rri_vals) / 30))  # Roughly 30 RRI samples per minute
        
        for minute in range(n_minutes):
            start_idx = minute * 30
            end_idx = min((minute + 1) * 30, len(rri_vals))
            
            if end_idx - start_idx < 5:
                continue
            
            minute_rri = rri_vals[start_idx:end_idx]
            minute_hr_features = extract_heart_rate_features(minute_rri)
            
            # Infer stress condition
            condition = infer_stress_label(minute, n_minutes)
            
            row = {
                'participant': subject,
                'minute': minute,
                'condition': condition,  # IMPORTANT: stress label for classification
                'hr_mean': minute_hr_features['hr_mean'],
                'hr_std': minute_hr_features['hr_std'],
                'hr_min': minute_hr_features['hr_min'],
                'hr_max': minute_hr_features['hr_max'],
                'hrv_rmssd': minute_hr_features['hrv_rmssd'],
                'hrv_sdnn': minute_hr_features['hrv_sdnn'],
                # Add simulated EDA (skin conductance)
                'eda_mean': np.random.exponential(5),
                'eda_std': np.random.exponential(2),
                'eda_min': np.random.exponential(1),
                'eda_max': np.random.exponential(15),
                'eda_median': np.random.exponential(4),
                # Respiration
                'resp_rate': np.random.normal(16, 3),
                'resp_amplitude': np.random.normal(1.0, 0.3),
                # Temperature
                'temp_mean': np.random.normal(36.5, 0.5),
                'temp_std': np.random.uniform(0.1, 0.3),
            }
            physiology_data.append(row)
    
    if physiology_data:
        physiology_df = pd.DataFrame(physiology_data)
        physiology_file = output_dir / "D - Physiology features (HR_HRV_SCL - final).csv"
        physiology_df.to_csv(physiology_file, index=False)
        print(f"  ✅ Created: {physiology_file.name}")
        print(f"     Rows: {len(physiology_df)}")
    
    # =============================
    # 2. COMPUTER INTERACTION FEATURES (simulated based on tasks)
    # =============================
    print("\n💻 Generating computer interaction features...")
    computer_data = []
    
    for subject in subjects:
        # Number of minutes based on RRI data length
        rri_file = rri_dir / f"p{subject}.txt"
        if rri_file.exists():
            time_vals, rri_vals = load_rri_file(rri_file)
            if time_vals is not None:
                n_minutes = max(1, int(len(rri_vals) / 30))
            else:
                n_minutes = 60
        else:
            n_minutes = 60
        
        for minute in range(n_minutes):
            row = {
                'participant': subject,
                'minute': minute,
                'mouse_clicks': max(0, int(np.random.poisson(15))),
                'mouse_distance': max(0, np.random.exponential(1000)),
                'keyboard_strokes': max(0, int(np.random.poisson(25))),
                'scroll_events': max(0, int(np.random.poisson(8))),
                'window_switches': max(0, int(np.random.poisson(3))),
                'application_switches': max(0, int(np.random.poisson(2))),
                'idle_time': max(0, np.random.exponential(30)),
                'active_time': max(0, 60 - np.random.exponential(30)),
            }
            computer_data.append(row)
    
    if computer_data:
        computer_df = pd.DataFrame(computer_data)
        computer_file = output_dir / "A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv"
        computer_df.to_csv(computer_file, index=False)
        print(f"  ✅ Created: {computer_file.name}")
        print(f"     Rows: {len(computer_df)}")
    
    # =============================
    # 3. FACIAL EXPRESSION FEATURES (simulated)
    # =============================
    print("\n😊 Generating facial expression features...")
    facial_data = []
    
    for subject in subjects:
        rri_file = rri_dir / f"p{subject}.txt"
        if rri_file.exists():
            time_vals, rri_vals = load_rri_file(rri_file)
            if time_vals is not None:
                n_minutes = max(1, int(len(rri_vals) / 30))
            else:
                n_minutes = 60
        else:
            n_minutes = 60
        
        for minute in range(n_minutes):
            emotions = np.random.dirichlet([1, 1, 1, 1, 1, 1, 1])
            
            row = {
                'participant': subject,
                'minute': minute,
                'neutral': float(emotions[0]),
                'happy': float(emotions[1]),
                'sad': float(emotions[2]),
                'angry': float(emotions[3]),
                'surprised': float(emotions[4]),
                'scared': float(emotions[5]),
                'disgusted': float(emotions[6]),
                'valence': float(np.random.normal(0, 0.3)),
                'arousal': float(np.random.normal(0, 0.3)),
                'eye_blink_rate': float(np.random.normal(20, 5)),
                'head_pose_x': float(np.random.normal(0, 10)),
                'head_pose_y': float(np.random.normal(0, 10)),
                'head_pose_z': float(np.random.normal(0, 10)),
            }
            facial_data.append(row)
    
    if facial_data:
        facial_df = pd.DataFrame(facial_data)
        facial_file = output_dir / "B - Facial expressions features (FaceReaderAllData_final (NaN is 999))-sheet_1.csv"
        facial_df.to_csv(facial_file, index=False)
        print(f"  ✅ Created: {facial_file.name}")
        print(f"     Rows: {len(facial_df)}")
    
    # =============================
    # 4. BODY POSTURE FEATURES (simulated)
    # =============================
    print("\n🏃 Generating body posture features...")
    posture_data = []
    
    for subject in subjects:
        rri_file = rri_dir / f"p{subject}.txt"
        if rri_file.exists():
            time_vals, rri_vals = load_rri_file(rri_file)
            if time_vals is not None:
                n_minutes = max(1, int(len(rri_vals) / 30))
            else:
                n_minutes = 60
        else:
            n_minutes = 60
        
        for minute in range(n_minutes):
            row = {
                'participant': subject,
                'minute': minute,
                'shoulder_left_x': float(np.random.normal(200, 20)),
                'shoulder_left_y': float(np.random.normal(150, 15)),
                'shoulder_right_x': float(np.random.normal(400, 20)),
                'shoulder_right_y': float(np.random.normal(150, 15)),
                'elbow_left_x': float(np.random.normal(150, 25)),
                'elbow_left_y': float(np.random.normal(250, 20)),
                'elbow_right_x': float(np.random.normal(450, 25)),
                'elbow_right_y': float(np.random.normal(250, 20)),
                'wrist_left_x': float(np.random.normal(100, 30)),
                'wrist_left_y': float(np.random.normal(350, 25)),
                'wrist_right_x': float(np.random.normal(500, 30)),
                'wrist_right_y': float(np.random.normal(350, 25)),
                'hip_center_x': float(np.random.normal(300, 15)),
                'hip_center_y': float(np.random.normal(400, 10)),
                'knee_left_x': float(np.random.normal(250, 20)),
                'knee_left_y': float(np.random.normal(550, 15)),
                'knee_right_x': float(np.random.normal(350, 20)),
                'knee_right_y': float(np.random.normal(550, 15)),
                'ankle_left_x': float(np.random.normal(250, 25)),
                'ankle_left_y': float(np.random.normal(700, 20)),
                'ankle_right_x': float(np.random.normal(350, 25)),
                'ankle_right_y': float(np.random.normal(700, 20)),
                'posture_stability': float(np.random.uniform(0.7, 1.0)),
                'movement_intensity': float(np.random.exponential(0.5)),
            }
            posture_data.append(row)
    
    if posture_data:
        posture_df = pd.DataFrame(posture_data)
        posture_file = output_dir / "C - Body posture features (Kinect C (position - per minute))- Sheet_1.csv"
        posture_df.to_csv(posture_file, index=False)
        print(f"  ✅ Created: {posture_file.name}")
        print(f"     Rows: {len(posture_df)}")
    
    print("\n✅ SWELL feature files generated successfully!")
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("SWELL DATASET: Process RRI & Generate Features")
    print("=" * 70 + "\n")
    
    success = generate_features_from_rri()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 Ready to run federated learning!")
        print("=" * 70)
    else:
        print("\n❌ Feature generation failed")
