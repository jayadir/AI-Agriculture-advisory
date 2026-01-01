from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from app.models.base import PyObjectId
from app.models.user import Location



class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatSession(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_phone: str
    messages: List[Message] = []
    summary: Optional[str] = None # For quick history lookup
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True

class AlertReport(BaseModel):
    """Used when a farmer reports a pest/disease"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    issue: str
    severity: Literal["low", "medium", "high"]
    location: Location # Required for geospatial clustering
    reported_by: str # User Phone
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True