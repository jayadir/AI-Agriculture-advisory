from fastapi import APIRouter,Depends,HTTPException,BackgroundTasks
from pydantic import BaseModel
from app.db.mongodb import get_database
from app.models.webhook import SMSPayload
from app.rag.engine import get_rag_engine,RAGEngine
from app.models.user import UserInDB
from app.models.chat import Message
from datetime import datetime, timezone
from bson import ObjectId
from app.services.web_pipeline import invoke_web_pipeline
from app.agents.react_agent_v2.graph import chat_with_agent
from app.agents.react_agent_v2.learning import learn_from_session
from typing import Optional
from app.services.llm import generate_response, classify_data_quality

router=APIRouter()

@router.post("/webhook", response_description="Handle Incoming SMS/IVR")
async def handle_gsm_webhook(
    payload: SMSPayload,
    background_tasks: BackgroundTasks,
    db = Depends(get_database),
    engine: RAGEngine = Depends(get_rag_engine)
):
    
    
    phone = payload.sender
    query_text = payload.text
    
    user = await db["users"].find_one({"phone_number": phone})
    
    if not user:
        print(f"New Farmer detected: {phone}")
        new_user = UserInDB(phone_number=phone, full_name="Guest Farmer")
        await db["users"].insert_one(new_user.model_dump(by_alias=True))
    
    # Check if there's an existing chat session for this phone number
    existing_session = await db["chat_sessions"].find_one(
        {"user_phone": phone},
        sort=[("updated_at", -1)]  # Get most recent session
    )
    
    thread_id = None
    chat_history = []
    
    if existing_session:
        thread_id = existing_session.get("thread_id")
        # Get last 10 messages (only user queries and AI responses)
        messages = existing_session.get("messages", [])
        # Filter and get last 10 user-assistant pairs
        chat_history = messages[-10:] if len(messages) > 10 else messages
        print(f"Continuing existing chat session with thread_id: {thread_id}")
    else:
        print(f"Starting new chat session for {phone}")
    
    # Pass explicit chat history for more control over context
    answer_text = chat_with_agent(phone, query_text, thread_id, chat_history)
    thread_id = answer_text.get("thread_id")
    
    # Save/update chat session in database
    await save_chat_session(db, phone, thread_id, query_text, answer_text.get("response", ""))
    
    background_tasks.add_task(
        learn_from_session,
        thread_id,
    )

    return answer_text


async def save_chat_session(db, phone: str, thread_id: str, query: str, response: str):
    """
    Save or update chat session with new message.
    """
    try:
        user_msg = Message(role="user", content=query, timestamp=datetime.now(timezone.utc))
        assistant_msg = Message(role="assistant", content=response, timestamp=datetime.now(timezone.utc))
        
        # Check if session exists
        existing_session = await db["chat_sessions"].find_one(
            {"user_phone": phone},
            sort=[("updated_at", -1)]
        )
        
        if existing_session and existing_session.get("thread_id") == thread_id:
            # Update existing session
            await db["chat_sessions"].update_one(
                {"_id": existing_session["_id"]},
                {
                    "$push": {
                        "messages": {
                            "$each": [user_msg.model_dump(), assistant_msg.model_dump()]
                        }
                    },
                    "$set": {
                        "updated_at": datetime.now(timezone.utc),
                        "summary": query[:50]  # Update summary with latest query
                    }
                }
            )
            print(f"Updated chat session for {phone}")
        else:
            # Create new session
            session_dict = {
                "_id": ObjectId(),
                "user_phone": phone,
                "thread_id": thread_id,
                "messages": [user_msg.model_dump(), assistant_msg.model_dump()],
                "summary": query[:50],
                "updated_at": datetime.now(timezone.utc)
            }
            
            await db["chat_sessions"].insert_one(session_dict)
            print(f"Created new chat session for {phone}")
            
    except Exception as e:
        print(f"Error saving chat session: {e}")


async def log_interaction(db, phone, query, response):
    """Helper to save chat history (deprecated - use save_chat_session instead)"""
    session_data = {
        "user_phone": phone,
        "messages": [
            {"role": "user", "content": query, "timestamp": datetime.now(timezone.utc)},
            {"role": "assistant", "content": response, "timestamp": datetime.now(timezone.utc)}
        ],
        "summary": query[:50],
        "updated_at": datetime.now(timezone.utc)
    }
    await db["chat_sessions"].insert_one(session_data)


