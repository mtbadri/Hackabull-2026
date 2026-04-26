---
inclusion: always
---

# Coding Standards

## Python
- Python 3.10+ — use `match/case`, `X | Y` union types, `list[dict]` generics
- All env vars loaded via `python-dotenv` (`load_dotenv()`) or `pydantic-settings`
- Never hardcode API keys, URIs, or credentials — always use `.env`
- Use `logging` (not `print`) for all runtime output
- Wrap all external API calls (Gemini, ElevenLabs, MongoDB, Snowflake) in try/except with graceful fallback
- Use `pathlib.Path` over `os.path`

## FastAPI (AI Brain)
- Use `pydantic` models for all request/response bodies
- Async handlers with `async def` + `await`
- Return structured JSON matching the shared event schema

## Streamlit (Dashboard)
- `st.set_page_config()` must be the first Streamlit call
- Use `st.session_state` for caching between reruns
- Always show a warning banner when a data source is unavailable

## File Structure
- Each service is self-contained under `services/{service}/`
- Shared constants/types go in a `services/shared/` module if needed
- Known faces: `services/vision/known_faces/{name}.jpg` + `{name}.json`

## Dependencies
- All packages pinned in `requirements.txt`
- Do not add new packages without updating `requirements.txt`
