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

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple/ \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --timeout 300 \
    --retries 5 \
    -r requirements.txt

COPY app/ ./app/

ENV PYTHONPATH=/workspace

EXPOSE 8000
EXPOSE 8501
