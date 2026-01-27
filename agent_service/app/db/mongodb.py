import asyncio
import os
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "agri_brain_db")

class Database:
    client: Optional[AsyncIOMotorClient] = None

db = Database()
_client_lock = asyncio.Lock()


async def _ensure_connected() -> None:
    if db.client is not None:
        return
    async with _client_lock:
        if db.client is not None:
            return
        await connect_to_mongo()

async def get_database():
    await _ensure_connected()
    return db.client[DB_NAME]  

async def connect_to_mongo():
    if not MONGO_URL:
        raise RuntimeError(
            "MONGO_URL is missnig"
        )

    client = AsyncIOMotorClient(MONGO_URL)
    try:
        await client.admin.command("ping")
    except Exception as exc:
        client.close()
        raise RuntimeError(
            f"Failed to connect to MongoDB at '{MONGO_URL}'. "
        ) from exc

    db.client = client
    print("Connected to MongoDB")

async def close_mongo_connection():
    """Closes connection (Call this on Shutdown)"""
    if db.client:
        db.client.close()
        print("MongoDB Connection Closed")