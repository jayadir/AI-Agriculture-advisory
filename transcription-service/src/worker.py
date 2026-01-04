import boto3
import json
import time
import os
import urllib.parse
from dotenv import load_dotenv
from src.asr_engine import CodeSwitchASR

load_dotenv()
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
REGION = os.getenv("AWS_REGION", "us-east-1")
TEMP_DIR = "tmp"

os.makedirs(TEMP_DIR, exist_ok=True)

sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

def main():
    print("[Worker] Initializing AI Models...")
    asr_engine = CodeSwitchASR()
    print("[Worker] Models Ready. Waiting for SQS messages...")

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
            )

            if "Messages" not in response:
                continue

            message = response["Messages"][0]
            receipt_handle = message["ReceiptHandle"]

            print("\n[Worker] Message received!")

            body = json.loads(message["Body"])

            if "Records" not in body:
                print("[Worker] Skipping non-S3 event.")
                sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
                continue

            s3_record = body["Records"][0]["s3"]
            bucket_name = s3_record["bucket"]["name"]
            file_key = urllib.parse.unquote_plus(s3_record["object"]["key"])

            print(f"[Worker] Processing: {file_key} from {bucket_name}")

            local_filename = os.path.join(TEMP_DIR, os.path.basename(file_key))
            s3.download_file(bucket_name, file_key, local_filename)

            transcription_text = asr_engine.process_file(local_filename)
            print(f"[Worker] Transcription Result: {transcription_text[:50]}...")

            # txt_key = "transcriptions/"+file_key + ".txt"
            # s3.put_object(
            #     Bucket=bucket_name,
            #     Key=txt_key,
            #     Body=transcription_text,
            #     ContentType="text/plain",
            # )
            
            # print(f"[Worker] Uploaded text to: {txt_key}")

            ai_agent_payload = {
                "transcription": transcription_text,
                "original_file": file_key,
                "bucket": bucket_name,
                "caller-number":s3.head_object(Bucket=bucket_name, Key=file_key)['Metadata']['caller-number'],
                "source":"call"
            }
            sqs.send_message(
                QueueUrl=os.getenv("AGENT_JOBS_QUEUE_URL"),
                MessageBody=json.dumps(ai_agent_payload)
            )
            print(f"[Worker] Sent text to AI Agent Queue")
            if os.path.exists(local_filename):
                os.remove(local_filename)

            sqs.delete_message(
                QueueUrl=SQS_QUEUE_URL,
                ReceiptHandle=receipt_handle,
            )
            print("[Worker] Job Done. Waiting...")

        except Exception as e:
            print(f"[Worker] Error: {str(e)}")
            time.sleep(5)

if __name__ == "__main__":
    main()
