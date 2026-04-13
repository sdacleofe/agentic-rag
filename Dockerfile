FROM python:3.11-slim

WORKDIR /workspace

# 🔥 Replace Debian mirror with direct CDN (covers legacy .list and DEB822 .sources)
RUN find /etc/apt -name "*.list" -o -name "*.sources" | \
    xargs sed -i 's|http://deb.debian.org|http://cdn-fastly.deb.debian.org|g' 2>/dev/null || true

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Install heavy packages in separate layers so Docker can cache each one
RUN pip install --no-cache-dir --prefer-binary --timeout 300 --retries 10 \
    "sentence-transformers>=3.0.0"

RUN pip install --no-cache-dir --prefer-binary --timeout 300 --retries 10 \
    "chromadb>=0.5.0"

RUN pip install --no-cache-dir --prefer-binary --timeout 300 --retries 10 \
    "streamlit>=1.38.0"

RUN pip install --no-cache-dir --prefer-binary --timeout 300 --retries 10 \
    "langgraph>=0.2.0"

# Install remaining packages (heavy ones above are already cached)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary --timeout 300 --retries 10 \
    -r requirements.txt

COPY app/ ./app/

ENV PYTHONPATH=/workspace

EXPOSE 8000
EXPOSE 8501
