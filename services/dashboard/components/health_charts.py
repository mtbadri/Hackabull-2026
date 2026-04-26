import pandas as pd
import plotly.express as px
import streamlit as st


def render_health_chart(df: pd.DataFrame | None, snowflake_error: bool):
    if snowflake_error:
        st.info("Snowflake unavailable — chart not available")
        fig = px.line(title="Health Events — Last 24 Hours")
        st.plotly_chart(fig, use_container_width=True)
        return

    if df is not None and not df.empty:
        fig = px.line(df, x="hour", y="count", title="Health Events — Last 24 Hours")
        st.plotly_chart(fig, use_container_width=True)
