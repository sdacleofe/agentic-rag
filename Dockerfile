FROM python:3.11-slim

WORKDIR /workspace

# 🔥 Replace Debian mirror with direct CDN (no redirect)
RUN sed -i 's|http://deb.debian.org|http://cdn-fastly.deb.debian.org|g' /etc/apt/sources.list

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV PYTHONPATH=/workspace

EXPOSE 8000
EXPOSE 8501
