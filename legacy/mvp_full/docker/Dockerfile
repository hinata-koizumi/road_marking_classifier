FROM nvidia/cuda:12.3.0-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential cmake git python3 python3-venv python3-pip python3-dev \
    libpdal-dev pdal libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip && pip install -r requirements.txt

COPY . .
CMD ["python3", "scripts/extract_and_vectorize.py", "--help"]
