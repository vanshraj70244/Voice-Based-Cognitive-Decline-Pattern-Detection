import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
from project import compute_risk_score
# ... rest of your imports



import streamlit as st
from project import compute_risk_score
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

# Title
st.title("Voice-Based Cognitive Decline Detection")
st.write("Upload an audio file to analyze speech patterns and estimate potential cognitive risk indicators.")

# File uploader
audio_file = st.file_uploader("D:\VSCODE\cv-corpus-20.0-delta-2024-12-06-en\cv-corpus-20.0-delta-2024-12-06\en\clips\common_voice_en_41247167.mp3", type=["wav", "mp3"])

# When a file is uploaded
if audio_file is not None:
    # Save the uploaded file temporarily
    with open("common_voice_en_41247167.mp3", "wb") as f:
        f.write(audio_file.read())

    st.audio("common_voice_en_41247167.mp3", format="audio/mp3")

    # Run analysis
    try:
        st.info("Analyzing audio. This may take a few seconds...")
        results = compute_risk_score("common_voice_en_41247167.mp3")

        # Show transcription
        st.subheader("📝 Transcription")
        st.write(results["transcription"])

        # Show features
        st.subheader("📊 Extracted Features")
        features_to_show = {k: v for k, v in results.items() if k != "transcription"}
        st.dataframe(pd.DataFrame([features_to_show]))

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
