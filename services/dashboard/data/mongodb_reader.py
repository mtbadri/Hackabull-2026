import certifi
from pymongo import MongoClient
from services.dashboard.settings import get_settings

_cached_data: list[dict] = []


def get_mongo_client():
    settings = get_settings()
    client = MongoClient(settings.MONGODB_URI, tlsCAFile=certifi.where())
    return client


def fetch_latest_events(n: int = 50) -> tuple[list[dict], bool]:
    global _cached_data
    settings = get_settings()
    try:
        client = get_mongo_client()
        collection = client[settings.MONGODB_DB][settings.MONGODB_COLLECTION]
        docs = list(
            collection.find({}, {"_id": 0})
            .sort("processed_at", -1)
            .limit(n)
        )
        _cached_data = docs
        return (docs, False)
    except Exception:
        return (_cached_data, True)
