import streamlit as st
import joblib
import librosa
import numpy as np

# Load pre-trained models
gender_model = joblib.load('gender_model.pkl')
emotion_model = joblib.load('emotion_model.pkl')

# Define feature extraction function
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file_name, sr=None)
    features = np.array([])

    if mfcc:
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        features = np.hstack((features, mfccs_mean))

    if chroma:
        stft = np.abs(librosa.stft(X))
        chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_mean = np.mean(chroma_stft.T, axis=0)
        features = np.hstack((features, chroma_mean))

    if mel:
        mel_spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        mel_mean = np.mean(mel_spectrogram.T, axis=0)
        features = np.hstack((features, mel_mean))

    return features

# Streamlit application layout
st.title("Speech Emotion and Gender Recognition")
st.write("Upload an audio file to predict emotion and gender.")

# File uploader
audio_file = st.file_uploader("Choose an audio file", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    # Extract features from audio file
    features = extract_feature("temp_audio.wav").reshape(1, -1)

    # Predict gender
    gender_prediction = gender_model.predict(features)[0]
    st.write(f"Predicted Gender: {gender_prediction}")

    # Predict emotion
    emotion_prediction = emotion_model.predict(features)[0]
    st.write(f"Predicted Emotion: {emotion_prediction}")

    # Cleanup temporary file
    import os
    os.remove("temp_audio.wav")
