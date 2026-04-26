import sys
from pydantic_settings import BaseSettings


class DashboardSettings(BaseSettings):
    MONGODB_URI: str
    MONGODB_DB: str
    MONGODB_COLLECTION: str
    PATIENT_NAME: str

    # Snowflake (optional — skipped for now)
    SNOWFLAKE_ACCOUNT: str = ""
    SNOWFLAKE_USER: str = ""
    SNOWFLAKE_PASSWORD: str = ""
    SNOWFLAKE_DATABASE: str = ""
    SNOWFLAKE_SCHEMA: str = ""
    SNOWFLAKE_WAREHOUSE: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


def get_settings() -> DashboardSettings:
    try:
        return DashboardSettings()
    except Exception as e:
        print(f"[ERROR] Missing required environment variables: {e}")
        sys.exit(1)
