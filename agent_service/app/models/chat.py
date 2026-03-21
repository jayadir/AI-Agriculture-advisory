from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from app.models.base import PyObjectId
from app.models.user import Location


class Turn(BaseModel):
    """A single query/response exchange."""
    user_query: str
    ai_response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatHistory(BaseModel):
    """
    Stores the full conversation history for a user, keyed by phone number.
    One document per user in the `chat_history` collection.
    """
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    phone_number: str = Field(..., description="User phone number (primary key)")
    turns: List[Turn] = []
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True


# Keep Message as an alias so any other code that imports it doesn't break immediately.
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

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