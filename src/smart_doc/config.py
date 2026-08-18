"""Central configuration and constants."""

# Supported file types
SUPPORTED_FILE_TYPES = ["pdf", "txt", "docx"]

# RAG Defaults
CHUNK_SIZE_DEFAULT = 1000
CHUNK_OVERLAP_DEFAULT = 200
RETRIEVER_K_DEFAULT = 5

# NVIDIA NIM Configuration
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"

# Available NIM Chat Models: (display_name, model_id)
# Source: https://build.nvidia.com/models (auto-updated catalog)
NIM_CHAT_MODELS = [
    ("NVIDIA Nemotron 3 Ultra 550B (Terkuat)", "nvidia/nemotron-3-ultra-550b-a55b"),
    ("NVIDIA Nemotron 3.5 Lightning 30B (Seimbang)", "nvidia/nemotron-3.5-lightning-30b-a3b"),
    ("Meta Muse Glimmer 30B (Kreatif)", "meta/muse-glimmer-30b"),
    ("StepFun Step 3.7 Flash (Cepat)", "stepfun-ai/step-3.7-flash"),
    ("Poolside Laguna XS 2.1 (Ringan)", "poolside/laguna-xs-2.1"),
    ("Thinking Machines Inkling", "thinkingmachines/inkling"),
]
