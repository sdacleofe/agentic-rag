import os

# LLM
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama3.2")

# Embedding & reranking
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# Storage
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./data/chromadb")
UPLOADS_PATH: str = os.getenv("UPLOADS_PATH", "./data/uploads")
BM25_INDEX_PATH: str = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")

# Chunking — CHUNK_SIZE is in words (≈ 512 tokens at 1.3 tokens/word)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))

# Ollama generation options
NUM_PREDICT: int = int(os.getenv("NUM_PREDICT", "512"))
NUM_CTX: int = int(os.getenv("NUM_CTX", "4096"))
NUM_THREAD: int = int(os.getenv("NUM_THREAD", "4"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
