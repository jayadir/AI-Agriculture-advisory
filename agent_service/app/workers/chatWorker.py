import asyncio
import json
import boto3
import os
import logging
from datetime import datetime, timezone
from bson import ObjectId

from app.db.mongodb import get_database
from app.db.chat_history import get_recent_turns, append_turn
from app.models.user import UserInDB
from app.workers.registration_node import is_registration_message, register_user_from_sms

AGENT_VERSION = os.getenv("AGENT_VERSION", "deep_research")

if AGENT_VERSION == "deep_research":
    from app.agents.deep_research_agent.graph import chat_with_agent
    from app.agents.deep_research_agent.learning import learn_from_session
else:
    from app.agents.react_agent_v2.graph import chat_with_agent
    from app.agents.react_agent_v2.learning import learn_from_session

SQS_QUEUE_URL = os.getenv("AGENT_JOBS_QUEUE_URL")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AGENT_RESPONSE_QUEUE_URL = os.getenv(
    "AGENT_RESPONSE_QUEUE_URL",
    "https://sqs.ap-south-1.amazonaws.com/963716652927/agent-response-queue",
)
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI_Worker")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)


def publish_agent_response(phone: str, thread_id: str, response_text: str, source: str):
    """Publish generated responses to a queue for downstream delivery handlers."""
    try:
        sqs = boto3.client("sqs", region_name=AWS_REGION)
        payload = {
            "phone": phone,
            "thread_id": thread_id,
            "response": response_text,
            "source": source,
            "status": "generated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        sqs.send_message(
            QueueUrl=AGENT_RESPONSE_QUEUE_URL,
            MessageBody=json.dumps(payload),
        )
        logger.info(f"Published response to agent-response queue for {phone}")
    except Exception as e:
        logger.error(f"Failed to publish response to agent-response queue: {e}")

# ---------------------------------------------------------------------------
# Twilio client — used only for call delivery (not SMS).
# ---------------------------------------------------------------------------
def _get_twilio_client():
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    if not sid or not token:
        logger.warning("Twilio credentials not set (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN). "
                       "Call delivery will be skipped.")
        return None
    try:
        from twilio.rest import Client
        return Client(sid, token)
    except Exception as e:
        logger.warning(f"Could not initialise Twilio client: {e}")
        return None

async def process_message(message, db):
    try:
        payload = json.loads(message)
        phone = payload.get("caller-number") or payload.get("phone")
        query = payload.get("transcription") or payload.get("text") or payload.get("body")
        source = payload.get("source", "sms").lower()   # "sms" | "call"

        logger.info(f"[{source.upper()}] Incoming message from {phone}: {query}")

        if not phone or not query:
            logger.warning("Invalid message payload — missing phone or query text.")
            return

        # ---- Registration node (SMS only) ------------------------------------
        if source == "sms" and is_registration_message(query):
            registered = await register_user_from_sms(db, phone, query)
            if registered:
                publish_agent_response(phone, "", "ok", source)
                logger.info(f"[SMS] Registration completed for {phone}; sent OK ack.")
            else:
                logger.warning(f"[SMS] Registration format matched but registration failed for {phone}.")
            return

        # ---- Load last 10 turns for context ----------------------------------
        chat_history = await get_recent_turns(db, phone, limit=10)
        if chat_history:
            logger.info(f"Loaded {len(chat_history) // 2} previous turns for {phone}")
        else:
            logger.info(f"No prior history found for {phone} — starting fresh")

        # ---- Run agent -------------------------------------------------------
        result = chat_with_agent(phone, query, chat_history=chat_history)
        response_text = result.get("response", "")

        logger.info(f"[FINAL ANSWER] To {phone}:\n{response_text}")

        # ---- Persist turn ---------------------------------------------------
        await append_turn(db, phone, query, response_text)
        logger.info(f"Appended turn to chat history for {phone}")

        # ---- Queue response for downstream delivery -------------------------
        publish_agent_response(phone, "", response_text, source)

        # ---- Deliver response -----------------------------------------------
        if source == "sms":
            logger.info(
                f"[SMS] Response queued to agent-response queue for {phone}; "
                "Twilio SMS sending is disabled."
            )
        elif source == "call":
            trigger_call(phone, response_text)
        else:
            logger.warning(f"Unknown source '{source}' — response not delivered.")

        # ---- Background learning --------------------------------------------
        asyncio.create_task(learn_from_session(phone))
        logger.info("Background learning task created")

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)


def trigger_call(to_number: str, text_response: str):
    """Initiate an outbound call that reads the agent response aloud via Twilio."""
    # TODO: integrate call delivery once Twilio credentials are configured
    client = _get_twilio_client()
    if not client:
        logger.info(f"[CALL PLACEHOLDER] To: {to_number} | Response: {text_response[:80]}...")
        return
    twiml = f"""
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
        call = client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            twiml=twiml
        )
        logger.info(f"Call initiated to {to_number}, SID: {call.sid}")
    except Exception as e:
        logger.error(f"Failed to trigger call to {to_number}: {e}")       
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