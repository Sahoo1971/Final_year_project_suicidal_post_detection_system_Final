import streamlit as st
import tensorflow as tf
import os
import gdown

# Model path and download details
LOCAL_MODEL_PATH = "model.h5"
GOOGLE_DRIVE_FILE_ID = "1TpquJore9fUncY-dj55msAIruNO0av51"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

# Download from Google Drive if model not present
def download_model_from_gdrive():
    if not os.path.exists(LOCAL_MODEL_PATH):
        try:
            st.info("Downloading model from Google Drive...")
            gdown.download(DOWNLOAD_URL, LOCAL_MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()
    else:
        st.info("Model already exists locally.")

# Load model and cache for performance
@st.cache_resource
def load_model():
    try:
        download_model_from_gdrive()
        model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
