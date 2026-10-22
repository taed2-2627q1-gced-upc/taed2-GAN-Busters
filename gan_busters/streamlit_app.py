import os
import streamlit as st

st.set_page_config(
    page_title="GAN-Busters",
    layout="wide",
)

API_URL = os.getenv(
    "GAN_BUSTERS_API_URL",
    "http://0.0.0.0:8081",
).rstrip("/")

st.title("GAN-Busters")
st.caption("AI-generated image detection platform")

st.markdown(
    """
    Use the sidebar to navigate between:

    - **Statistics:** API performance and MLflow training metrics.
    - **Predict:** Upload JPEG or PNG images and inspect predictions.
    """
)

st.info(
    f"API endpoint: {API_URL}\n\n"
    "Open Statistics to inspect service performance, or Predict "
    "to classify images."
)
