import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "agri_brain_db"

class Database:
    client: AsyncIOMotorClient = None

db = Database()

async def get_database():
    """Dependency to get the database instance."""
    return db.client[DB_NAME]

async def connect_to_mongo():
    """Connects to MongoDB (Call this on Startup)"""
    db.client = AsyncIOMotorClient(MONGO_URL)
    print("Connected to MongoDB")

async def close_mongo_connection():
    """Closes connection (Call this on Shutdown)"""
    if db.client:
        db.client.close()
        print("MongoDB Connection Closed")