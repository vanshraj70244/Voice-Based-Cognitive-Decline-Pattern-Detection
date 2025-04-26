# Voice-Based Cognitive Decline Detection

# 1. Imports and Setup
import os
import numpy as np
import librosa
import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from pydub import AudioSegment
AudioSegment.converter = "ffmpeg"  # 👈 Add this line immediately after importing AudioSegment

from collections import Counter
# import IPython.display as ipd
import time

# Handle whisper import compatibility
try:
    import whisper
    model_loader = whisper.load_model
except (AttributeError, ImportError):
    model_loader = None
    print("Could not load Whisper model. Check installation.")

nltk.download('punkt')
# nltk.download('punkt_tab')  # ❌ Commented: 'punkt_tab' does NOT exist in NLTK (this line causes error)

# ❌ spacy.cli.download("en_core_web_sm")  # Cannot download on Streamlit Cloud (Permission Denied)
nlp = spacy.load("en_core_web_sm")  # ✅ Just load the model directly

# 2. Audio Preprocessing
def preprocess_audio(path):
    sound = AudioSegment.from_file(path)
    sound = sound.set_channels(1).set_frame_rate(16000)
    temp_path = "temp.wav"
    sound.export(temp_path, format="wav")
    y, sr = librosa.load(temp_path)
    return y, sr

# 3. Transcription using Whisper
def transcribe_audio(path):
    if model_loader is None:
        raise ImportError("Whisper model is not loaded properly.")
    model = model_loader("base")
    result = model.transcribe(path)
    return result["text"]

# 4. Feature Extraction
## 4.1 Audio Features
def extract_audio_features(y, sr):
    duration = librosa.get_duration(y=y, sr=sr)
    energy = librosa.feature.rms(y=y)[0]
    silence = energy < 0.01
    pause_ratio = np.sum(silence) / len(energy)
    return {"duration": duration, "pause_ratio": pause_ratio}

## 4.2 Textual Features
def extract_text_features(text):
    tokens = nltk.word_tokenize(text.lower())
    total = len(tokens)
    unique = len(set(tokens))
    hesitation_words = ["uh", "um", "eh", "er"]
    hesitations = sum([text.lower().count(w) for w in hesitation_words])
    return {
        "total_words": total,
        "unique_words": unique,
        "lexical_diversity": unique / total if total > 0 else 0,
        "hesitations": hesitations
    }

# 5. Risk Scoring Function
def compute_risk_score(audio_path):
    y, sr = preprocess_audio(audio_path)
    text = transcribe_audio(audio_path)
    audio_feats = extract_audio_features(y, sr)
    text_feats = extract_text_features(text)
    features = {**audio_feats, **text_feats, "transcription": text}
    return features

# 6. Modeling and Analysis
def train_model(features_list):
    df = pd.DataFrame(features_list)
    if df.empty:
        raise ValueError("No valid features to train the model. Dataset may be empty.")
    model = IsolationForest(contamination=0.2)
    model.fit(df.drop(columns=["transcription", "filename"], errors='ignore'))
    scores = model.decision_function(df.drop(columns=["transcription", "filename"], errors='ignore'))
    df["anomaly_score"] = -scores
    return df, model

# 7. Visualization
def visualize_features(df):
    sns.pairplot(df.drop(columns=["transcription", "filename"], errors='ignore'))
    plt.show()

# 8. Batch Processing Dataset Folder
def process_large_dataset(folder_path, limit=None):
    files = [f for f in os.listdir(folder_path) if f.endswith('.mp3') or f.endswith('.wav')]
    if limit:
        files = files[:limit]

    features_list = []
    for idx, file in enumerate(files):
        path = os.path.join(folder_path, file)
        try:
            print(f"Processing {idx+1}/{len(files)}: {file}")
            features = compute_risk_score(path)
            features["filename"] = file
            features_list.append(features)
        except Exception as e:
            print(f"Error processing {file}: {e}")
        time.sleep(0.5)

    print(f"\nTotal successfully processed files: {len(features_list)}")
    return features_list

# 9. Main Execution
if __name__ == "__main__":
    dataset_folder = "D:/VSCODE/cv-corpus-20.0-delta-2024-12-06-en/cv-corpus-20.0-delta-2024-12-06/en/clips"  # Update this to your correct dataset path

    print("\nStarting batch processing...")
    features_list = process_large_dataset(dataset_folder)

    if len(features_list) == 0:
        print("No features extracted. Please check audio format or folder path.")
        exit()

    df, model = train_model(features_list)

    print("\nSample of extracted features with anomaly scores:")
    print(df.head())

    visualize_features(df)

    # Optional: Save the features to CSV
    df.to_csv("audio_features_with_risk_scores.csv", index=False)
    print("\nSaved features with risk scores to audio_features_with_risk_scores.csv")
