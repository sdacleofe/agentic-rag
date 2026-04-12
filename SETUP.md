# Setup Guide — Business Insights Agentic RAG

Two environments are covered:

| Environment | Hardware | Model | GPU |
|---|---|---|---|
| **Dev (laptop)** | i9 2.2GHz · 8GB RAM · 16GB VGPU | `gemma4:e4b` (8B) | NVIDIA (full offload) |
| **Production (Azure)** | Standard_D4s_v3 · 4 vCPU · 16GB RAM | `gemma4:e2b` (5B) | CPU only |

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone & configure](#2-clone--configure)
3. [Dev setup (GPU laptop)](#3-dev-setup-gpu-laptop)
4. [Production setup (Azure VM)](#4-production-setup-azure-vm)
5. [First use — ingest documents & query](#5-first-use--ingest-documents--query)
6. [Service URLs](#6-service-urls)
7. [Day-to-day commands](#7-day-to-day-commands)
8. [Configuration reference](#8-configuration-reference)
9. [Hardening for production](#9-hardening-for-production)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

### Both environments

- **Docker Desktop ≥ 25** (Windows/Mac) or **Docker Engine ≥ 25 + Docker Compose v2** (Linux)
- **Git**
- **`make`** — comes with Git for Windows (Git Bash) or WSL2; on Linux: `sudo apt install make`

### Dev laptop only (GPU)

- NVIDIA GPU driver installed — confirm with:
  ```bash
  nvidia-smi
  ```
- **NVIDIA Container Toolkit** — required for Docker to access the GPU:
  ```bash
  # Ubuntu / Debian / WSL2
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- Confirm Docker can see the GPU:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
  ```

> **Windows note:** Run all commands in **WSL2** (Ubuntu). Docker Desktop with WSL2 backend supports NVIDIA GPU passthrough automatically once the driver is installed on the Windows host — you do **not** need to separately install the driver inside WSL2.

---

## 2. Clone & configure

```bash
git clone <your-repo-url> agentic-rag
cd agentic-rag
```

The repository already contains `.env` (production) and `.env.dev` (dev/GPU). Review and adjust if needed:

```bash
# Production defaults (Azure CPU)
cat .env

# Dev defaults (GPU laptop — gemma4:e4b, 8192 ctx, 8 threads)
cat .env.dev
```

No secrets are required — the stack is fully local/self-hosted.

---

## 3. Dev setup (GPU laptop)

### 3.1 — First-time setup (one command)

```bash
make dev-setup
```

This will:
1. Build the Python application Docker image
2. Start the Ollama container with GPU passthrough
3. Pull `gemma4:e4b` (~5.2GB download — runs once, cached in a Docker volume)
4. Start the API, UI, and all services

Expected output when complete:
```
Dev services started:
  Streamlit UI  → http://localhost:8501
  FastAPI docs  → http://localhost:8000/docs
  Ollama API    → http://localhost:11434
```

### 3.2 — Verify GPU is being used

```bash
# In a second terminal, watch VGPU utilisation while sending a test query
watch -n1 nvidia-smi

# Send a test prompt to Ollama directly
curl http://localhost:11434/api/generate \
  -d '{"model":"gemma4:e4b","prompt":"What is GDP?","stream":false}' \
  | python3 -m json.tool | grep response
```

You should see ~30–50 tokens/sec and VGPU utilisation spike during generation.

### 3.3 — Memory fit (your hardware)

| Component | VGPU | RAM |
|---|---|---|
| Gemma 4 E4B weights (Q4_K) | ~5.2 GB | — |
| KV cache @ 8192 ctx | ~2.0 GB | — |
| Embedding model (bge-small-en-v1.5) | — | ~90 MB |
| Re-ranker (bge-reranker-base) | — | ~350 MB |
| ChromaDB + FastAPI + Streamlit | — | ~500 MB |
| **Total** | **~7.2 GB / 16 GB** | **~1 GB / 8 GB** |

### 3.4 — Live code reload

The dev compose mounts `./app` into the container. Any change to Python files in `app/` is picked up immediately by the FastAPI `--reload` server. Streamlit also auto-reloads on file save.

---

## 4. Production setup (Azure VM)

### 4.1 — Provision the VM

In the Azure portal or CLI:

```bash
az vm create \
  --resource-group myRG \
  --name rag-prod \
  --image Ubuntu2204 \
  --size Standard_D4s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# Open HTTP and HTTPS ports
az vm open-port --port 80 --resource-group myRG --name rag-prod
az vm open-port --port 443 --resource-group myRG --name rag-prod
```

### 4.2 — Install Docker on the VM

```bash
ssh azureuser@<VM_PUBLIC_IP>

# Install Docker Engine
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose v2 plugin
sudo apt-get install -y docker-compose-plugin

# Verify
docker compose version
```

### 4.3 — Clone the project

```bash
git clone <your-repo-url> agentic-rag
cd agentic-rag

# Confirm .env has production values (CPU model, etc.)
cat .env
```

### 4.4 — First-time setup (one command)

```bash
make setup
```

This will:
1. Build the application image on the VM
2. Pull `gemma4:e2b` (~3.2GB, CPU-optimised Q4_K)
3. Start all 4 services: Ollama, API, UI, Nginx

The UI will be available at `http://<VM_PUBLIC_IP>` once complete.

### 4.5 — Auto-restart on reboot

```bash
sudo tee /etc/systemd/system/rag.service > /dev/null <<'EOF'
[Unit]
Description=Agentic RAG Stack
After=docker.service
Requires=docker.service

[Service]
WorkingDirectory=/home/azureuser/agentic-rag
ExecStart=/usr/bin/docker compose up
ExecStop=/usr/bin/docker compose down
Restart=on-failure
User=azureuser

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rag
```

---

## 5. First use — ingest documents & query

### Via the Streamlit UI

1. Open the UI (see [Service URLs](#6-service-urls))
2. In the **left sidebar**, click **Browse files** and upload one or more PDFs
3. Wait for the "Ingested N chunks" confirmation
4. Type a question into the chat input — e.g. *"What are the key revenue drivers mentioned in the Q3 report?"*

### Via the API directly

```bash
# Upload a PDF
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@./my-report.pdf"

# Simple synchronous query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the cash flow outlook?"}'

# Complex async query
TASK=$(curl -s -X POST http://localhost:8000/query/async \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare revenue trends across all uploaded reports and identify risk factors"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])")

# Poll for result
curl http://localhost:8000/query/status/$TASK | python3 -m json.tool

# List ingested documents
curl http://localhost:8000/documents/list | python3 -m json.tool
```

### Interactive API docs

FastAPI auto-generates a Swagger UI at `http://localhost:8000/docs` — you can test all endpoints directly in the browser.

---

## 6. Service URLs

| Service | Dev (laptop) | Production (Azure) |
|---|---|---|
| Streamlit Chat UI | http://localhost:8501 | http://\<VM_IP\> |
| FastAPI (REST) | http://localhost:8000 | http://\<VM_IP\>/api |
| FastAPI Swagger docs | http://localhost:8000/docs | — (not exposed in prod) |
| Ollama | http://localhost:11434 | internal only |

---

## 7. Day-to-day commands

```bash
# ── Dev (GPU laptop) ─────────────────────────────────────────────────────
make dev-start          # start all services
make dev-stop           # stop all services
make dev-logs           # follow all logs
make dev-logs-api       # follow API logs only
make dev-logs-ui        # follow UI logs only

# ── Production (Azure) ───────────────────────────────────────────────────
make start              # start all services
make stop               # stop all services
make restart            # rolling restart
make logs               # follow all logs
make logs-api           # follow API logs only

# ── Rebuild after code changes ───────────────────────────────────────────
make dev-build          # rebuild image (dev)
make build              # rebuild image (prod)
```

---

## 8. Configuration reference

All settings are controlled via environment variables. Set them in `.env` (prod) or `.env.dev` (dev).

| Variable | Default (prod) | Default (dev) | Description |
|---|---|---|---|
| `MODEL_NAME` | `gemma4:e2b` | `gemma4:e4b` | Ollama model tag |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | `http://localhost:11434` | Ollama endpoint |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | same | HuggingFace embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | same | HuggingFace cross-encoder |
| `CHUNK_SIZE` | `400` | same | Words per chunk (≈512 tokens) |
| `CHUNK_OVERLAP` | `50` | same | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `20` | same | Candidates before re-ranking |
| `RERANK_TOP_K` | `5` | same | Chunks sent to LLM |
| `NUM_CTX` | `4096` | `8192` | Ollama context window size |
| `NUM_THREAD` | `4` | `8` | CPU threads for LLM |
| `NUM_PREDICT` | `512` | `1024` | Max output tokens |
| `TEMPERATURE` | `0.1` | `0.1` | Generation temperature |
| `CHROMA_PATH` | `/data/chromadb` | `./data/chromadb` | ChromaDB persistence path |
| `UPLOADS_PATH` | `/data/uploads` | `./data/uploads` | Uploaded PDF storage path |

---

## 9. Hardening for production

### TLS / HTTPS (strongly recommended)

```bash
# On the Azure VM
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
sudo systemctl reload nginx
```

### Authentication

The Streamlit UI has no built-in auth. Add Nginx basic auth as a quick guard:

```bash
sudo apt install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd youruser
```

Then add to the `location /` block in `nginx.conf`:
```nginx
auth_basic "Restricted";
auth_basic_user_file /etc/nginx/.htpasswd;
```

### Firewall (Azure NSG)

Ensure only these inbound ports are open:

| Port | Protocol | Purpose |
|---|---|---|
| 22 | TCP | SSH (restrict to your IP) |
| 80 | TCP | HTTP (redirect to HTTPS) |
| 443 | TCP | HTTPS |

Ports 8000, 8501, and 11434 should **not** be open to the public internet.

---

## 10. Troubleshooting

### `nvidia-smi` works but Docker can't see the GPU

```bash
# Re-run container toolkit config and restart Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Confirm
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

### Model download is slow / fails

```bash
# Pull the model manually and check progress
docker exec -it ollama ollama pull gemma4:e4b
```

### "No relevant documents found" on every query

```bash
# Check collection size
curl http://localhost:8000/documents/list

# Check ChromaDB directly
docker exec -it rag_api python3 -c \
  "from app.ingest import get_chroma_collection; c=get_chroma_collection(); print(c.count(), 'chunks')"
```

### API times out on complex queries

Increase `proxy_read_timeout` in `nginx.conf` (default 120s) and restart Nginx:
```bash
make restart
```

### Out of memory (OOM) on laptop

Reduce context window in `.env.dev`:
```
NUM_CTX=4096
```
Then `make dev-stop && make dev-start`.

### Port already in use

```bash
# Find and kill the conflicting process
sudo lsof -i :8501
sudo kill -9 <PID>
```
