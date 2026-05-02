# AI-Agriculture-advisory (major_proj)

This repo runs the agriculture advisory system using **workers** (you can ignore the FastAPI apps if you’re not using APIs).

## Repo layout

- `agent_service/` — SQS-based agent worker + RAG/LLM agents + MongoDB chat history.
- `transcription-service/` — SQS-based transcription worker (S3 → ASR → agent jobs SQS).
- `s3-upload-service/` — (Optional) Node webhook server + Redis/Bull worker (Twilio → S3/SQS).

## Prerequisites

- Windows + PowerShell
- Python (recommended: 3.12 for `agent_service`)
- Node.js 18+ (only if using `s3-upload-service`)
- MongoDB (local or Atlas)
- Redis (only if using `s3-upload-service`)
- AWS: SQS queues + S3 bucket for the pipeline



# 1) agent_service (worker)

## 1.1 Environment variables (`agent_service/.env`)

Minimum to run the worker:

```dotenv
# MongoDB
MONGO_URL=mongodb://localhost:27017
DB_NAME=agri_brain_db

# Optional: vector DB path
VECTOR_DB_PATH=artifacts/vector_db/agri_faiss_index

# LLM
LLM_PROVIDER=GROQ           # GROQ | GOOGLE | OLLAMA
LLM_MODEL=
GROQ_API_KEY=
GOOGLE_API_KEY=

# Optional web search
TAVILY_API_KEY=
TAVILY_MAX_RESULTS=10

# SQS queues
AWS_REGION=ap-south-1
AGENT_JOBS_QUEUE_URL=
AGENT_RESPONSE_QUEUE_URL=

# Agent selection
AGENT_VERSION=deep_research  # deep_research | react_agent_v2

# Optional reranker switches
ENABLE_RERANKER=1
RERANKER_MODEL_NAME=JAYADIR/mdts-agxqa-circuit-full-bm25
RERANKER_BATCH_SIZE=16
RERANKER_TOP_K=7

# Optional
FORCE_CPU=0
```

Notes:
- `.env` is loaded via `python-dotenv`.
- If you hit `ModuleNotFoundError: pydantic_settings`, install `pydantic-settings`.

## 1.2 Install + run

```powershell
cd agent_service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pydantic-settings

python app\workers\chatWorker.py
```

---

# 2) transcription-service (worker)

## 2.1 Environment variables (`transcription-service/.env`)

```dotenv
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# SQS input (S3 event notifications typically point here)
SQS_QUEUE_URL=

# Downstream queue for agent jobs
AGENT_JOBS_QUEUE_URL=
```

## 2.2 Install + run

```powershell
cd transcription-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# If you get `ModuleNotFoundError: torch` or `torchaudio`, install PyTorch + torchaudio appropriate for your OS/CUDA.
# (On many GPU runtimes, these are preinstalled.)

python -m src.worker
```

## 2.3 Run the transcription worker on a cloud GPU (Runpod / Google Colab)

The transcription worker does not expose an HTTP server—it's a long-running process that polls `SQS_QUEUE_URL` and reads audio from S3.

Required environment variables (same as local):

```dotenv
AWS_REGION=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
SQS_QUEUE_URL=
AGENT_JOBS_QUEUE_URL=
```

Also required: PyTorch + torchaudio (the worker imports `torch` and `torchaudio`). Many GPU images (Runpod/Colab) already include them.

### Option A: Runpod (recommended for always-on worker)

1) Create a GPU Pod using a PyTorch/CUDA image.

2) Set the env vars above in the Pod template (preferred), or export them in the terminal.

3) In the Pod terminal:

```bash
cd /workspace
git clone https://github.com/jayadir/AI-Agriculture-advisory.git
cd AI-Agriculture-advisory/transcription-service

pip install -r requirements.txt

# If you hit audio decode errors, install ffmpeg:
# apt-get update && apt-get install -y ffmpeg

python -m src.worker
```

### Option B: Google Colab (good for quick tests)

1) In Colab: Runtime → Change runtime type → GPU.

2) Run in a cell:

```bash
!git clone https://github.com/jayadir/AI-Agriculture-advisory.git
%cd AI-Agriculture-advisory/transcription-service

!pip install -r requirements.txt

# If torchaudio is missing, install PyTorch + torchaudio matching the Colab CUDA runtime.
# (Colab often already has torch installed.)

import os
os.environ["AWS_REGION"] = "ap-south-1"
os.environ["AWS_ACCESS_KEY_ID"] = "..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "..."
os.environ["SQS_QUEUE_URL"] = "..."
os.environ["AGENT_JOBS_QUEUE_URL"] = "..."
```

3) Then run the worker:

```bash
!python -m src.worker
```

---

# 3) s3-upload-service (optional)

Used if you want Twilio webhooks + upload-to-S3 + queueing.

## 3.1 Environment variables (`s3-upload-service/.env`)

```dotenv
PORT=3000

# Redis (Bull)
REDIS_HOST=127.0.0.1
REDIS_PORT=6379

# AWS
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
S3_BUCKET_NAME=

# Twilio recording fetch (used by upload worker)
TWILIO_API_KEY_SID=
TWILIO_API_KEY_SECRET=

# Agent queues
AGENT_JOBS_QUEUE_URL=
AGENT_RESPONSE_QUEUE_URL=
AGENT_RESPONSE_QUEUE_NAME=agent-response-queue
```

## 3.2 Run

Start Redis (local install) so it listens on `REDIS_HOST:REDIS_PORT`.

```powershell
cd s3-upload-service
npm install
npm run server
```

In a second terminal:

```powershell
cd s3-upload-service
npm run worker
```

## 3.3 Expose `s3-upload-service` to Twilio using ngrok

Twilio must reach your webhook over public HTTPS. ngrok provides a public HTTPS URL that tunnels to your local Node server.

1) Install and authenticate ngrok (once):

```powershell
ngrok version
ngrok config add-authtoken <YOUR_NGROK_TOKEN>
```

2) Start your Node server:

```powershell
cd s3-upload-service
npm run server
```

3) In a second terminal, start the tunnel:

```powershell
ngrok http 3000
```

Copy the **Forwarding** HTTPS URL, e.g. `https://<id>.ngrok-free.app`.

4) Configure Twilio webhooks (Twilio Console → Phone Numbers → your number):

- **Voice** → “A Call Comes In” (Webhook, `POST`):
	- `https://<your-ngrok-domain>/api/voice`
- **Messaging** → “A Message Comes In” (Webhook, `POST`):
	- `https://<your-ngrok-domain>/api/sms/webhook`

Notes:
- Your voice flow records audio and Twilio will post the recording to `POST /api/twilio/webhook` automatically (it’s referenced in the TwiML returned by `/api/voice`).
- Free ngrok URLs change when you restart it. If you want a stable URL, use a reserved domain in ngrok.

---

# End-to-end flow (high level)

1. (Optional) `s3-upload-service` receives Twilio webhook and uploads recording to S3 / forwards SMS to `AGENT_JOBS_QUEUE_URL`.
2. S3 event notification sends a message to the transcription SQS queue (`SQS_QUEUE_URL`).
3. `transcription-service` worker reads SQS, downloads audio from S3, transcribes, and sends text to `AGENT_JOBS_QUEUE_URL`.
4. `agent_service` worker reads agent jobs, runs the agent, stores chat history in MongoDB, and publishes the response to `AGENT_RESPONSE_QUEUE_URL`.

---

# Troubleshooting

- **MongoDB connection fails**: verify `MONGO_URL` and that MongoDB is running.
- **SQS/S3 permission errors**: ensure the AWS credentials (or IAM role) can access the relevant SQS queues and S3 bucket.
- **Redis connection errors** (optional service): ensure Redis is running and `REDIS_HOST`/`REDIS_PORT` are correct.
- **`pydantic_settings` missing**: run `pip install pydantic-settings` in `agent_service` venv.
