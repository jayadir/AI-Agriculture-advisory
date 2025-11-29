from fastapi import APIRouter,Depends,HTTPException,BackgroundTasks
from app.db.mongodb import get_database
from app.models.webhook import SMSPayload
from app.rag.engine import get_rag_engine,RAGEngine
from app.models.user import UserInDB
from datetime import datetime

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
    
    result = await engine.process(query_text)
    answer_text = result
    
    background_tasks.add_task(
        log_interaction, 
        db, 
        phone, 
        query_text, 
        answer_text
    )

    return answer_text


async def log_interaction(db, phone, query, response):
    """Helper to save chat history"""
    session_data = {
        "user_phone": phone,
        "messages": [
            {"role": "user", "content": query, "timestamp": datetime.utcnow()},
            {"role": "assistant", "content": response, "timestamp": datetime.utcnow()}
        ],
        "summary": query[:50],
        "updated_at": datetime.utcnow()
    }
    await db["chat_sessions"].insert_one(session_data)