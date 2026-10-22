"""Streamlit page for uploading images and displaying REAL/FAKE predictions."""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv(
    "GAN_BUSTERS_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

st.set_page_config(
    page_title="Predict | GAN-Busters",
    layout="wide",
)

st.title("Image Prediction")
st.caption("Classify images as REAL or FAKE")

st.write(
    "Upload one or more JPEG or PNG images. "
    "The FastAPI backend validates the files and runs inference."
)

uploaded_files = st.file_uploader(
    "Choose images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

st.caption("The API enforces its own file count, size and image dimension limits.")

if uploaded_files:
    st.subheader("Selected images")

    for uploaded in uploaded_files:
        st.image(
            uploaded.getvalue(),
            caption=uploaded.name,
            width=250,
        )

    if st.button(
        "Classify images",
        type="primary",
        use_container_width=True,
    ):
        files = [
            (
                "files",
                (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "application/octet-stream",
                ),
            )
            for uploaded in uploaded_files
        ]

        try:
            with st.spinner("Running model inference..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=180,
                )

            if not response.ok:
                try:
                    detail = response.json().get(
                        "detail",
                        response.text,
                    )
                except ValueError:
                    detail = response.text

                st.error(f"Prediction failed ({response.status_code}): {detail}")
                st.stop()

            payload = response.json()
            results = payload.get("predictions", [])

            if not results:
                st.warning("The API returned no predictions.")
                st.stop()

            st.success(f"Classified {len(results)} image(s).")
            st.subheader("Results")

            df = pd.DataFrame(results)
            df = df.drop(labels="fake_probability", axis=1)

            # Expected fields from ImagePrediction:
            # filename, label, fake_probability, confidence_score
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
            )

            for result in results:
                label = str(result["label"]).upper()
                filename = result["filename"]
                fake_probability = float(result["fake_probability"])
                confidence = float(result["confidence_score"])

                with st.container(border=True):
                    st.write(f"**{filename}**")
                    st.write(f"Classification: **{label}**")
                    st.progress(
                        min(1.0, max(0.0, confidence)),
                        text=f"Confidence: {confidence:.1%}",
                    )

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                data=csv,
                file_name="gan_busters_predictions.csv",
                mime="text/csv",
            )

        except requests.Timeout:
            st.error(
                "The prediction request timed out. Try a smaller batch or check the API logs."
            )
        except requests.RequestException as exc:
            st.error(f"Could not reach the prediction API: {exc}")
        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"Unexpected prediction response: {exc}")
else:
    st.info("Select at least one image to get started.")
