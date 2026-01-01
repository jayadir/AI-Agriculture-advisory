from pydantic import BaseModel,Field
class SMSPayload(BaseModel):
    
    sender:str=Field(...,description="Phone number of the sender")
    text:str=Field(...,description="Content of the SMS message")
    message_id:str=Field(default="unknown",description="Unique ID of the message")
    
    class Config:
        populate_by_name = True