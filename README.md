<p align="center">
  <a href="https://your-deployed-streamlit-link.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-blue?style=for-the-badge&logo=streamlit" alt="Live Demo Badge"/>
  </a>
</p>

---

# 🧠 Voice-Based Cognitive Decline Detection

A **Streamlit web application** to analyze speech patterns from audio files to detect early signs of cognitive decline based on **audio features** and **text features**.  
The system uses a combination of **deep learning (Whisper)** for transcription and **machine learning (Isolation Forest)** for risk analysis.

---

## 📸 Demo

👉 [Live App on Streamlit Cloud](https://voice-based-cognitive-decline-pattern-detection-tviavtsuu4fegb.streamlit.app/)  
---

## 🚀 Features

- 🎤 Upload `.wav` or `.mp3` audio files
- 📝 Get transcription of the speech using OpenAI's Whisper
- 📊 Extract important audio and textual features like pause ratio, lexical diversity, hesitation frequency, etc.
- 📈 Detect anomalies using Isolation Forest to estimate cognitive risk
- 🎯 Fast and interactive UI built with Streamlit
- 🌐 Deployed live using Streamlit Cloud

---

## 🛠 Technologies Used

- **Streamlit** – Frontend and backend framework
- **OpenAI Whisper** – Speech recognition
- **Librosa** – Audio processing
- **NLTK** – Natural Language Processing
- **spaCy** – Text feature extraction
- **Isolation Forest (sklearn)** – Anomaly detection
- **Pydub** – Audio format conversion
- **FFmpeg** – Backend audio support

---

## 🧩 Project Structure

---

## 🛠 How to Run Locally

1. **Clone the repository**:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

sudo apt update
sudo apt install -y ffmpeg libsndfile1
pip install -r requirements.txt

git init
git remote add origin https://github.com/yourusername/voice-cognitive-decline.git
git add .
git commit -m "Initial Commit"
git push -u origin master

git clone https://github.com/yourusername/voice-cognitive-decline.git
cd voice-cognitive-decline
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
python main.py


