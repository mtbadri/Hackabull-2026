# Dashboard Lead — Task List
## Role 3: Caregiver Hub (Streamlit · Port 8501)

---

## Task 1: Project Setup
- [x] Create `services/dashboard/` directory
- [ ] Create `services/dashboard/app.py` as the Streamlit entry point
- [ ] Create `services/dashboard/settings.py` with `DashboardSettings` using `pydantic-settings`, loading all required env vars (`MONGODB_URI`, `MONGODB_DB`, `MONGODB_COLLECTION`, `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_WAREHOUSE`, `PATIENT_NAME`); exit with non-zero status if any are missing
- [ ] Create `services/dashboard/data/` directory with `__init__.py`
- [ ] Create `services/dashboard/data/mongodb_reader.py` for MongoDB query helpers
- [ ] Create `services/dashboard/data/snowflake_reader.py` for Snowflake query helpers
- [ ] Create `services/dashboard/components/` directory with `__init__.py`
- [ ] Create `services/dashboard/components/event_feed.py` for the live feed table component
- [ ] Create `services/dashboard/components/health_charts.py` for the Plotly chart component
- [ ] Confirm `streamlit`, `plotly`, `pandas`, `pymongo`, `snowflake-connector-python`, and `pydantic-settings` are present in `requirements.txt`

---

## Task 2: MongoDB Reader (`data/mongodb_reader.py`)
- [ ] Write `get_mongo_client()` using `MONGODB_URI` from `DashboardSettings`
- [ ] Write `fetch_latest_events(n=50)` — query `MONGODB_DB.MONGODB_COLLECTION`, sort by `processed_at` descending, return list of dicts
- [ ] Wrap in try/except — on failure return `(cached_data, True)` where the boolean is a `mongo_error` flag; on success return `(data, False)`

---

## Task 3: Snowflake Reader (`data/snowflake_reader.py`)
- [ ] Write `get_snowflake_conn()` using env vars: `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_WAREHOUSE`
- [ ] Write `fetch_health_trends()` running:
  ```sql
  SELECT DATE_TRUNC('hour', processed_at) AS hour, COUNT(*) AS count
  FROM events
  WHERE type = 'health'
    AND processed_at >= DATEADD(hour, -24, CURRENT_TIMESTAMP())
  GROUP BY 1
  ORDER BY 1
  ```
- [ ] Return `(pandas.DataFrame with columns hour/count, False)` on success; return `(None, True)` on failure where the boolean is a `snowflake_error` flag

---

## Task 4: Live Event Feed Component (`components/event_feed.py`)
- [ ] Write `render_event_feed(events: list[dict], mongo_error: bool)` function
- [ ] If `mongo_error` is True, show `st.warning("MongoDB unavailable — showing cached data")` above the table
- [ ] Build a `pandas.DataFrame` from the events list
- [ ] Display columns: `timestamp`, `type`, `subtype`, `confidence`, `verified`, `voice_script`, `processing_status`
- [ ] Apply row-level color coding via `df.style.apply()`:
  - Yellow (`#fff9c4`) for `type == "health"`
  - Green (`#c8e6c9`) for `type == "identity"`
- [ ] Render with `st.dataframe(styled_df, use_container_width=True)`

---

## Task 5: Health Trends Chart Component (`components/health_charts.py`)
- [ ] Write `render_health_chart(df, snowflake_error: bool)` function
- [ ] If `snowflake_error` is True, show `st.info("Snowflake unavailable — chart not available")` and render an empty placeholder chart
- [ ] If `df` is not None, render `px.line(df, x="hour", y="count", title="Health Events — Last 24 Hours")`
- [ ] Display with `st.plotly_chart(fig, use_container_width=True)`

---

## Task 6: Family Sync Sidebar (`app.py`)
- [ ] Add a `st.sidebar` section titled "Family Sync"
- [ ] Inputs:
  - `st.file_uploader("Upload photo", type=["jpg","jpeg","png"])`
  - `st.text_input("Name")` — used as filename slug
  - `st.text_input("Relationship")`
  - `st.text_area("Background")`
  - `st.text_area("Last Conversation")`
- [ ] On `st.button("Save")`:
  - Validate all fields are filled; show `st.error()` if any are missing
  - Save image bytes → `services/vision/known_faces/{name}.jpg`
  - Save JSON → `services/vision/known_faces/{name}.json` with fields: `name`, `relationship`, `background`, `last_conversation`
  - Show `st.success("Saved! Vision Engine will detect {name} on next startup.")`

---

## Task 7: Main App Layout & Auto-Refresh (`app.py`)
- [ ] Set `st.set_page_config(page_title="AuraGuard Caregiver Portal", layout="wide")`
- [ ] Load `DashboardSettings` at startup; exit with error if any required env var is missing
- [ ] Show header: `AuraGuard AI — Caregiver Portal` with patient name from `PATIENT_NAME` and last refresh timestamp
- [ ] Organize main content into two tabs: `Live Feed` | `Health Trends`
- [ ] In `Live Feed` tab: call `fetch_latest_events()`, cache result in `st.session_state`, call `render_event_feed()`
- [ ] In `Health Trends` tab: call `fetch_health_trends()`, cache result in `st.session_state`, call `render_health_chart()`
- [ ] Both data fetches happen on every refresh cycle (every 5 seconds)
- [ ] Implement 5-second auto-refresh using `st.session_state` timestamp + `time.sleep(5)` + `st.rerun()`

---

## Task 8: Run & Verify
- [ ] Run: `streamlit run services/dashboard/app.py --server.port 8501`
- [ ] Confirm live feed refreshes every 5 seconds
- [ ] Confirm yellow/green row color coding renders correctly
- [ ] Confirm Plotly chart renders, or placeholder shows when Snowflake is unreachable
- [ ] Confirm MongoDB warning banner shows when MongoDB is unreachable
- [ ] Confirm a test photo upload writes both `.jpg` and `.json` to `services/vision/known_faces/`
