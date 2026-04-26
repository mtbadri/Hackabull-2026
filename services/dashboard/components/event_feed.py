import pandas as pd
import streamlit as st

DISPLAY_COLUMNS = [
    "timestamp", "type", "subtype", "confidence",
    "verified", "voice_script", "processing_status",
]


def _row_color(row):
    if row.get("type") == "health":
        color = "#fff9c4"
    elif row.get("type") == "identity":
        color = "#c8e6c9"
    else:
        color = ""
    return [f"background-color: {color}" if color else "" for _ in row]


def render_event_feed(events: list[dict], mongo_error: bool):
    if mongo_error:
        st.warning("MongoDB unavailable — showing cached data")

    if not events:
        st.info("No events yet.")
        return

    df = pd.DataFrame(events)

    # Keep only the columns we care about (add missing ones as empty)
    for col in DISPLAY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[DISPLAY_COLUMNS]

    styled = df.style.apply(_row_color, axis=1)
    st.dataframe(styled, use_container_width=True)
