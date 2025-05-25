import streamlit as st
import tensorflow as tf
import os
import requests
from dotenv import load_dotenv

load_dotenv()

DROPBOX_DIRECT_URL = os.getenv("DROPBOX_DIRECT_URL")  # Must end with ?dl=1
LOCAL_MODEL_PATH = "model.h5"

def download_model_from_dropbox():
    if not os.path.exists(LOCAL_MODEL_PATH):
        try:
            st.info("Downloading model...")
            response = requests.get(DROPBOX_DIRECT_URL)
            response.raise_for_status()
            with open(LOCAL_MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()
    else:
        st.info("Model already exists locally.")

@st.cache_resource
def load_model():
    try:
        download_model_from_dropbox()
        model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
