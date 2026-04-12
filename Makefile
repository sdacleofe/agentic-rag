.PHONY: setup start stop restart logs build pull-model \
        dev-build dev-setup dev-start dev-stop dev-logs

DEV_COMPOSE = docker compose -f docker-compose.yml -f docker-compose.dev.yml

# ── Production (Azure CPU) ──────────────────────────────────────────────────
build:
	docker compose build

pull-model:
	docker compose up -d ollama
	@echo "Waiting for Ollama to become ready..."
	@until docker exec ollama ollama list > /dev/null 2>&1; do \
		echo "  ...still waiting"; sleep 3; \
	done
	@echo "Ollama is ready. Pulling gemma4:e2b ..."
	docker exec ollama ollama pull gemma4:e2b
	@echo "Model pull complete."

setup: build pull-model
	docker compose up -d
	@echo "Services started. UI at http://localhost"

start:
	docker compose up -d

stop:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

logs-%:
	docker compose logs -f $*

# ── Dev (local GPU laptop) ──────────────────────────────────────────────────
dev-build:
	$(DEV_COMPOSE) build

dev-pull-model:
	$(DEV_COMPOSE) up -d ollama
	@echo "Waiting for Ollama (GPU)..."
	@until docker exec ollama ollama list > /dev/null 2>&1; do \
		echo "  ...still waiting"; sleep 3; \
	done
	@echo "Pulling gemma4:e4b (8B model for 16GB VGPU)..."
	docker exec ollama ollama pull gemma4:e4b
	@echo "Model ready."

dev-setup: dev-build dev-pull-model
	$(DEV_COMPOSE) up -d
	@echo ""
	@echo "Dev services started:"
	@echo "  Streamlit UI  → http://localhost:8501"
	@echo "  FastAPI docs  → http://localhost:8000/docs"
	@echo "  Ollama API    → http://localhost:11434"

dev-start:
	$(DEV_COMPOSE) up -d

dev-stop:
	$(DEV_COMPOSE) down

dev-logs:
	$(DEV_COMPOSE) logs -f

dev-logs-%:
	$(DEV_COMPOSE) logs -f $*
