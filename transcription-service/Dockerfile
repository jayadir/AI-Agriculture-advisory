FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


RUN pip install --no-cache-dir --pre torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/nightly/cu126

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY preload.py .

RUN mkdir -p tmp tmp_lid

RUN python preload.py && rm preload.py


ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.worker"]
