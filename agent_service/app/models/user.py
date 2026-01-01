from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime
from app.models.base import PyObjectId
from bson import ObjectId

class Location(BaseModel):
    """GeoJSON format for MongoDB geospatial queries"""
    type: str = "Point"
    coordinates: List[float]

class UserBase(BaseModel):
    phone_number: str = Field(..., description="Unique mobile number")
    full_name: Optional[str] = None
    crops: List[str] = []
    location: Optional[Location] = None

class UserCreate(UserBase):
    pass

class UserInDB(UserBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}