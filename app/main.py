from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db.mongodb import connect_to_mongo, close_mongo_connection, get_database
from app.core.config import settings

from app.api.v1 import users 
from app.api.v1 import chat

async def create_indexes():
    db = await get_database()
    
    await db["users"].create_index("phone_number", unique=True)
    await db["users"].create_index([("location", "2dsphere")])
    
    await db["alerts"].create_index([("location", "2dsphere")])
    
    print("MongoDB Indexes Verified/Created")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    await create_indexes()
    yield
    await close_mongo_connection()

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.include_router(
    users.router, 
    prefix=f"{settings.API_V1_STR}/users", 
    tags=["Users"]
)

app.include_router(
    chat.router, 
    prefix=f"{settings.API_V1_STR}/chat", 
    tags=["Chat"]
)

@app.get("/")
def read_root():
    return {"message": "Agri-Brain System Online with GeospatialDB 🌍"}
