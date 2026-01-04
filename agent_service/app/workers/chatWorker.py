import asyncio
import json
import boto3
import os
import logging
from datetime import datetime

from app.db.mongodb import get_database
from app.models.user import UserInDB
from app.agents.react_agent_v2.graph import chat_with_agent

from twilio.rest import Client

SQS_QUEUE_URL=os.getenv("AGENT_JOBS_QUEUE_URL")
AWS_REGION=os.getenv("AWS_REGION", "us-east-1")
TWILLIO_PHONE_NUMBER=os.getenv("TWILLIO_PHONE_NUMBER")

twilio_client=Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger("AI_Worker")

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
        
        user=await db["users"].find_one({"phone_number":phone})
        if not user:
            logger.info(f"New Farmer detected: {phone}")
            new_user = UserInDB(phone_number=phone, full_name="Guest Farmer")
            await db["users"].insert_one(new_user.model_dump(by_alias=True))
            
        response_text=chat_with_agent(phone,query)
        thread_id=response_text.get("thread_id")
        if source=="sms":
            send_sms_reply(phone,response_text.get("response",""))
        else:
            trigger_call(phone,response_text.get("response",""))
        if thread_id:
            asyncio.create_task(learn_from_session(thread_id))
            logger.info("Background learning started")
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
            from_=TWILIO_PHONE_NUMBER,
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
            for message in response["Messages"]:
                receipt_handle = msg['ReceiptHandle']
                body = msg['Body']
                
                await process_message(body, db)

                sqs.delete_message(
                    QueueUrl=SQS_QUEUE_URL,
                    ReceiptHandle=receipt_handle
                )

        except Exception as e:
            logger.error(f"Critical Worker Loop Error: {e}")
            await asyncio.sleep(5) # Prevent CPU spike on loop error

if __name__ == "__main__":
    asyncio.run(main())