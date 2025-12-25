from pydantic import BaseModel, Field,HttpUrl
from typing import List, Optional,Literal
from datetime import datetime
from app.models.base import PyObjectId
from bson import ObjectId

class CandidateMetadata(BaseModel):
    source_url: str = Field(..., description="Original URL of the source")
    title: str = Field(default="Unknown", description="Page title")
    thread_id: str = Field(..., description="The chat session that found this info")
    document_id: str = Field(..., description="UUID connecting all chunks to one parent webpage")
    chunk_index: int = Field(default=0, description="Sequence order of this chunk")
    chunk_id: str = Field(..., description="Unique UUID for this specific text block")
    
class CandidateKnowledge(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    page_content: str = Field(..., min_length=50, description="The actual text chunk")
    metadata: CandidateMetadata
    status: Literal["pending", "added", "rejected"] = Field(default="pending")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}