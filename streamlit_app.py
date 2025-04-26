import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)


import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # Disable Streamlit's live reload to avoid torch error


import streamlit as st
import pandas as pd
from app import compute_risk_score

# --- Streamlit UI ---

# App title
st.title("🧠 Voice-Based Cognitive Decline Detection")
st.write(
    "Upload an audio file (.wav or .mp3) to analyze speech patterns "
    "and estimate cognitive decline risk indicators based on audio and text features."
)

# File uploader
audio_file = st.file_uploader("🎤 Upload an audio file", type=["wav", "mp3"])

# When a file is uploaded
if audio_file is not None:
    # Save the uploaded file temporarily
    with open("temp_input.wav", "wb") as f:
        f.write(audio_file.read())

    # Display audio player
    st.audio("temp_input.wav", format="audio/wav")

    # Run analysis with a loading spinner
    with st.spinner("🔎 Analyzing audio... please wait..."):
        try:
            results = compute_risk_score("temp_input.wav")

            # Success message
            st.success("✅ Analysis complete!")

            # Display transcription
            st.subheader("📝 Transcription")
            st.markdown(f"```{results['transcription']}```")

            # Display extracted features
            st.subheader("📊 Extracted Features")
            features_to_show = {k: v for k, v in results.items() if k != "transcription"}
            st.dataframe(pd.DataFrame([features_to_show]))

        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {e}")

# Footer
st.markdown("---")
st.caption("Developed by Vanshraj Singh Rathore 🚀")
