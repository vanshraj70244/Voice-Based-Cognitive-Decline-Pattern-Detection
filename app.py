import os
import numpy as np
import librosa
import nltk
import pandas as pd
import matplotlib.pyplot as plt
# remove seaborn since you’re not using it for plots now
from sklearn.ensemble import IsolationForest
from nltk.tokenize import word_tokenize
from pydub import AudioSegment
AudioSegment.converter = "ffmpeg"
from collections import Counter
import time
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── ADD THIS ─────────────────────────────────────────────────────────────
# ensure the Punkt tokenizer is present on every deployment
nltk.download('punkt', quiet=True)
# ────────────────────────────────────────────────────────────────────────────

# whisper import compatibility
try:
    import whisper
    model_loader = whisper.load_model
except (AttributeError, ImportError):
    model_loader = None
    print("Could not load Whisper model. Check installation.")

nlp = spacy.load("en_core_web_sm")


def preprocess_audio(path):
    sound = AudioSegment.from_file(path)
    sound = sound.set_channels(1).set_frame_rate(16000)
    temp_path = "temp.wav"
    sound.export(temp_path, format="wav")
    y, sr = librosa.load(temp_path, sr=16000)
    return y, sr

def transcribe_audio(path):
    if model_loader is None:
        raise ImportError("Whisper model is not loaded properly.")
    model = model_loader("base")
    result = model.transcribe(path)
    return result["text"]

def extract_audio_features(y, sr):
    duration = librosa.get_duration(y=y, sr=sr)
    energy = librosa.feature.rms(y=y)[0]
    silence = energy < 0.01
    pause_ratio = np.sum(silence) / len(energy)
    return {"duration": duration, "pause_ratio": pause_ratio}

def extract_text_features(text):
    tokens = word_tokenize(text.lower())
    total = len(tokens)
    unique = len(set(tokens))
    hesitations = sum(text.lower().count(w) for w in ["uh", "um", "eh", "er"])
    return {
        "total_words": total,
        "unique_words": unique,
        "lexical_diversity": unique / total if total else 0,
        "hesitations": hesitations
    }

def compute_risk_score(audio_path):
    y, sr = preprocess_audio(audio_path)
    text = transcribe_audio(audio_path)
    audio_feats = extract_audio_features(y, sr)
    text_feats = extract_text_features(text)
    return {**audio_feats, **text_feats, "transcription": text}

def train_model(features_list):
    df = pd.DataFrame(features_list)
    df.dropna(inplace=True)
    model = IsolationForest(contamination=0.2, random_state=0)
    X = df.drop(columns=["transcription", "filename"], errors='ignore')
    model.fit(X)
    df["anomaly_score"] = -model.decision_function(X)
    return df, model

def visualize_features(df):
    pd.plotting.scatter_matrix(df.drop(columns=["transcription", "filename"], errors='ignore'), figsize=(8,8))
    plt.show()

def process_large_dataset(folder_path, limit=None):
    # ensure you only match real .wav/.mp3 files
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.mp3', '.wav'))]
    if limit:
        files = files[:limit]

    features_list = []
    for idx, file in enumerate(files, 1):
        path = os.path.join(folder_path, file)
        try:
            print(f"Processing {idx}/{len(files)}: {file}")
            feats = compute_risk_score(path)
            feats["filename"] = file
            features_list.append(feats)
        except Exception as e:
            print(f"Error processing {file}: {e}")
        time.sleep(0.5)
    print(f"\nTotal processed: {len(features_list)}")
    return features_list

if __name__ == "__main__":
    dataset_folder = "D:/VSCODE/cv-corpus-20.0-delta-2024-12-06-en/cv-corpus-20.0-delta-2024-12-06/en/clips"
    features = process_large_dataset(dataset_folder)
    if not features:
        print("No features extracted. Check formats and paths.")
        exit()

    df, model = train_model(features)
    print(df.head())
    visualize_features(df)
    df.to_csv("audio_features_with_risk_scores.csv", index=False)
    print("Saved to audio_features_with_risk_scores.csv")
