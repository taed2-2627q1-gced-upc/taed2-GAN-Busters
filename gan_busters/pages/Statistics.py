import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv(
    "GAN_BUSTERS_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

st.set_page_config(
    page_title="Statistics | GAN-Busters",
    layout="wide",
)

st.title("API & Model Statistics")
st.caption("Runtime observability and MLflow experiment information")

if st.button("Refresh statistics"):
    st.rerun()

try:
    response = requests.get(
        f"{API_URL}/analytics/performance",
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

except requests.RequestException as exc:
    st.error(f"Could not retrieve API statistics: {exc}")
    st.stop()

api = data["api"]
requests_data = api["requests"]
latency = api["latency"]
predictions = api["predictions"]
system = api["system"]
model = data["model"]
mlflow = model["mlflow"]

# API overview
st.header("API performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Requests recorded",
    requests_data["total_recorded"],
)
c2.metric(
    "Requests/min",
    requests_data["throughput_rpm"],
)
c3.metric(
    "Error rate",
    f'{requests_data["error_rate_percent"]:.2f}%',
)
c4.metric(
    "Prediction batches",
    predictions["total_batches_recorded"],
)

st.subheader("Request latency")

all_latency = latency["all"]
predict_latency = latency["predictions"]

c1, c2, c3 = st.columns(3)
c1.metric("Average", f'{all_latency["avg_ms"]:.2f} ms')
c2.metric("P95", f'{all_latency["p95_ms"]:.2f} ms')
c3.metric("P99", f'{all_latency["p99_ms"]:.2f} ms')

st.write("**Prediction endpoint**")

c1, c2, c3 = st.columns(3)
c1.metric(
    "Average latency",
    f'{predict_latency["avg_ms"]:.2f} ms',
)
c2.metric("P95 latency", f'{predict_latency["p95_ms"]:.2f} ms')
c3.metric("P99 latency", f'{predict_latency["p99_ms"]:.2f} ms')

st.caption(
    "Latency is measured over retained request history, "
    "not a fixed time window."
)

# HTTP response codes
st.subheader("HTTP response codes")

status_counts = requests_data["status_counts"]

if status_counts:
    status_df = pd.DataFrame(
        [
            {"Status code": code, "Requests": count}
            for code, count in status_counts.items()
        ]
    ).sort_values("Status code")

    st.bar_chart(
        status_df,
        x="Status code",
        y="Requests",
    )
else:
    st.info("No requests recorded yet.")

# System metrics
st.subheader("System resources")

c1, c2, c3 = st.columns(3)
c1.metric("CPU utilization", f'{system["cpu_percent"]:.1f}%')
c2.metric("System memory", f'{system["memory_percent"]:.1f}%')
c3.metric("API process RAM", f'{system["process_memory_mb"]:.1f} MB')

# Inference batch statistics
st.subheader("Inference statistics")

c1, c2, c3 = st.columns(3)
c1.metric(
    "Images processed",
    predictions["total_images_recorded"],
)
c2.metric(
    "Average batch size",
    predictions["average_batch_size"],
)
c3.metric(
    "Maximum batch size",
    predictions["max_batch_size"],
)

# Model status
st.header("Model information")

c1, c2 = st.columns(2)
c1.metric(
    "Model loaded",
    "Yes" if model["loaded"] else "No",
)
c2.metric("Prediction threshold", model["threshold"])

st.subheader("MLflow")

if not mlflow["available"]:
    st.warning(
        mlflow.get(
            "error",
            "MLflow information is unavailable.",
        )
    )
else:
    run = mlflow.get("latest_run")
    registered = mlflow.get("registered_model")

    if run:
        st.write("**Latest training run**")
        st.json({
            "run_id": run["run_id"],
            "status": run["status"],
            "start_time": run["start_time"],
            "end_time": run["end_time"],
        })

        metrics = run.get("metrics", {})
        if metrics:
            st.write("**Training metrics**")
            metrics_df = pd.DataFrame(
                [
                    {"Metric": key, "Value": value}
                    for key, value in metrics.items()
                ]
            )
            st.dataframe(metrics_df, hide_index=True)

        with st.expander("Training parameters and tags"):
            st.json({
                "params": run.get("params", {}),
                "tags": run.get("tags", {}),
            })
    else:
        st.info("No training runs found in the configured experiment.")

    if registered:
        st.write("**Registered model**")
        st.json(registered)
    else:
        st.info("No registered model version was found.")
