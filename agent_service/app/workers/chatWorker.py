import asyncio
import json
import boto3
import os
import logging
from datetime import datetime, timezone
from bson import ObjectId

from app.db.mongodb import get_database
from app.models.user import UserInDB
from app.agents.react_agent_v2.graph import chat_with_agent
from app.agents.react_agent_v2.learning import learn_from_session
from twilio.rest import Client

SQS_QUEUE_URL=os.getenv("AGENT_JOBS_QUEUE_URL")
AWS_REGION=os.getenv("AWS_REGION", "us-east-1")
TWILLIO_PHONE_NUMBER=os.getenv("TWILLIO_PHONE_NUMBER")

twilio_client=Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger("AI_Worker")

async def save_chat_session(db, phone: str, thread_id: str, query: str, response: str):
    """
    Save or update chat session with new message.
    """
    try:
        from app.models.chat import Message
        
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
            logger.info(f"Updated chat session for {phone}")
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
            logger.info(f"Created new chat session for {phone}")
            
    except Exception as e:
        logger.error(f"Error saving chat session: {e}")

async def process_message(message,db):
    try:
        payload=json.loads(message)
        phone=payload.get("caller-number")
        query=payload.get("transcription")
        source=payload.get("source","sms")
        
        logger.info(f"Processing message from {phone}: {query}")
        
        if not phone or not query:
            logger.warning("Invalid message payload, missing phone or transcription.")
            return
        
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
            logger.info(f"Continuing existing chat session with thread_id: {thread_id}")
        else:
            logger.info(f"Starting new chat session for {phone}")
        
        # Pass explicit chat history for more control over context
        response_text = chat_with_agent(phone, query, thread_id, chat_history)
        thread_id = response_text.get("thread_id")
        
        # Save/update chat session in database
        await save_chat_session(db, phone, thread_id, query, response_text.get("response", ""))
        
        if source=="sms":
            send_sms_reply(phone,response_text.get("response",""))
        else:
            trigger_call(phone,response_text.get("response",""))
        if thread_id:
            asyncio.create_task(learn_from_session(thread_id))
            logger.info("Background learning started")
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        
def send_sms_reply(to_number,text):
    try:
        message=twilio_client.messages.create(
            body=text,
            from_=TWILLIO_PHONE_NUMBER,
            to=to_number
        )
        logger.info(f"Sent SMS to {to_number}, SID: {message.sid}")
    except Exception as e:
        logger.error(f"Failed to send SMS to {to_number}: {e}")
def trigger_call(to_number, text_response):
    print(f"   [Call] Initiating call to {to_number}...")
    

    twiml_instructions = f"""
    <Response>
        <Say voice="alice" language="en-IN">
            Hello. Here is the answer to your query.
            {text_response}
        </Say>
        <Pause length="1"/>
        <Say>Goodbye.</Say>
    </Response>
    """

    try:
        call = twilio_client.calls.create(
            to=to_number,
            from_=os.getenv("TWILIO_PHONE_NUMBER"),
            twiml=twiml_instructions
        )
        print(f"   [Call] Call started: {call.sid}")
    except Exception as e:
        print(f"   [Call] Failed to trigger call: {e}")       
async def main():
    sqs=boto3.client("sqs",region_name=AWS_REGION)
    db=await get_database()
    logger.info(f"AI Worker started. Listening on {SQS_QUEUE_URL}...")
    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                AttributeNames=['All']
            )
            if "Messages" not in response:
                continue
            for msg in response["Messages"]:
                receipt_handle = msg['ReceiptHandle']
                body = msg['Body']
                
                await process_message(body, db)

                sqs.delete_message(
                    QueueUrl=SQS_QUEUE_URL,
                    ReceiptHandle=receipt_handle
                )

        except Exception as e:
            logger.error(f"Critical Worker Loop Error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())