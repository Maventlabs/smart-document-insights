"""Central configuration and constants."""

# Supported file types
SUPPORTED_FILE_TYPES = ["pdf", "txt", "docx"]

# RAG Defaults
CHUNK_SIZE_DEFAULT = 1000
CHUNK_OVERLAP_DEFAULT = 200
RETRIEVER_K_DEFAULT = 5

# NVIDIA NIM Configuration
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"

# Available NIM Chat Models: (display_name, model_id)
NIM_CHAT_MODELS = [
    ("Meta Llama 3.1 8B (Cepat)", "meta/llama-3.1-8b-instruct"),
    ("Meta Llama 3.1 70B (Kuat)", "meta/llama-3.1-70b-instruct"),
    ("Meta Llama 3.3 70B (Terbaru)", "meta/llama-3.3-70b-instruct"),
    ("NVIDIA Nemotron 4 340B (Terkuat)", "nvidia/nemotron-4-340b-instruct"),
    ("Mistral Large 2 (Multilingual)", "mistralai/mistral-large-2-instruct"),
    ("DeepSeek R1 (Reasoning)", "deepseek/deepseek-r1"),
    ("Qwen 3 235B (Multilingual)", "qwen/qwen3-235b-a22b"),
]
