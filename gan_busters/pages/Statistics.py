
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

API_URL = os.getenv(
    "GAN_BUSTERS_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

HISTORY_LIMIT = 120
DEFAULT_REFRESH_SECONDS = 5

st.set_page_config(
    page_title="Statistics | GAN-Busters",
    layout="wide",
)

st.title("API & Model Statistics")
st.caption("Runtime observability and MLflow experiment information")

# Dashboard refresh controls
with st.sidebar:
    st.header("Dashboard settings")

    auto_refresh = st.toggle(
        "Auto-refresh",
        value=True,
    )

    refresh_seconds = st.selectbox(
        "Refresh interval",
        options=[2, 5, 10, 15, 30, 60],
        index=1,
        format_func=lambda seconds: f"{seconds} seconds",
    )

    if st.button("Refresh statistics now", use_container_width=True):
        st.rerun()

if auto_refresh:
    st_autorefresh(
        interval=refresh_seconds * 1000,
        limit=0,
        key="statistics_auto_refresh",
    )

# Initialize per-session resource history
if "resource_history" not in st.session_state:
    st.session_state.resource_history = []

# Fetch current API statistics
try:
    response = requests.get(
        f"{API_URL}/analytics/performance",
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

except requests.RequestException as exc:
    st.error(f"Could not retrieve API statistics: {exc}")
    st.caption(
        "Check that the API is running and that "
        "GAN_BUSTERS_API_URL points to the correct address."
    )
    st.stop()

api = data["api"]
requests_data = api["requests"]
latency = api["latency"]
predictions = api["predictions"]
system = api["system"]
model = data["model"]
mlflow = model["mlflow"]

# Record a resource sample on each successful refresh
sample = {
    "timestamp": datetime.now(),
    "CPU utilization (%)": float(system["cpu_percent"]),
    "System memory (%)": float(system["memory_percent"]),
    "API process RAM (MB)": float(system["process_memory_mb"]),
}

st.session_state.resource_history.append(sample)
st.session_state.resource_history = (
    st.session_state.resource_history[-HISTORY_LIMIT:]
)

st.caption(
    f"Last updated: {sample['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} "
    f"| Refresh interval: "
    f"{refresh_seconds}s" if auto_refresh else
    f"Last updated: {sample['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} "
    "| Auto-refresh paused"
)

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

# Request latency
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
c2.metric(
    "P95 latency",
    f'{predict_latency["p95_ms"]:.2f} ms',
)
c3.metric(
    "P99 latency",
    f'{predict_latency["p99_ms"]:.2f} ms',
)

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
            {"Status code": str(code), "Requests": count}
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

# System resource metrics
st.header("System resources")

c1, c2, c3 = st.columns(3)

c1.metric(
    "CPU utilization",
    f'{system["cpu_percent"]:.1f}%',
)
c2.metric(
    "System memory",
    f'{system["memory_percent"]:.1f}%',
)
c3.metric(
    "API process RAM",
    f'{system["process_memory_mb"]:.1f} MB',
)

# Historical resource charts
history_df = pd.DataFrame(
    st.session_state.resource_history
).set_index("timestamp")

st.subheader("Resource utilization over time")

st.caption(
    f"Showing the last {len(history_df)} samples. "
    "History is retained in this browser session and "
    "is reset when the session is lost."
)

# CPU utilization
st.write("**CPU utilization (%)**")
st.line_chart(
    history_df[["CPU utilization (%)"]],
    y="CPU utilization (%)",
    height=220,
)

# System memory utilization
st.write("**System memory utilization (%)**")
st.line_chart(
    history_df[["System memory (%)"]],
    y="System memory (%)",
    height=220,
)

# API process memory usage
st.write("**API process RAM (MB)**")
st.line_chart(
    history_df[["API process RAM (MB)"]],
    y="API process RAM (MB)",
    height=220,
)

# Optional combined chart for quick comparison
with st.expander("View all resource metrics together"):
    st.line_chart(
        history_df[
            [
                "CPU utilization (%)",
                "System memory (%)",
            ]
        ],
        height=300,
    )

# Inference batch statistics
st.header("Inference statistics")

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
c2.metric(
    "Prediction threshold",
    model["threshold"],
)

# MLflow
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
        st.json(
            {
                "run_id": run["run_id"],
                "status": run["status"],
                "start_time": run["start_time"],
                "end_time": run["end_time"],
            }
        )

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
            st.json(
                {
                    "params": run.get("params", {}),
                    "tags": run.get("tags", {}),
                }
            )
    else:
        st.info(
            "No training runs found in the configured experiment."
        )

    if registered:
        st.write("**Registered model**")
        st.json(registered)
    else:
        st.info("No registered model version was found.")
