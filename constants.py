# =============================================================================
# constants.py — System-wide constants
# =============================================================================

# Audio Settings
SAMPLE_RATE  = 16000   # 16 kHz -- universally supported by browsers & Deepgram STT
CHANNELS     = 1
SAMPLE_WIDTH = 2       # 16-bit PCM = 2 bytes per sample
CHUNK_FRAMES = 4096    # larger buffer = more stable streaming

# Scoring Weights
EXP_SIM_WEIGHT   = 0.35
EXP_STR_WEIGHT   = 0.40
SKILL_COV_WEIGHT = 0.25
DEFAULT_THRESHOLD = 0.50

# Models
EXTRACTION_MODEL   = "minimax-m2.7:cloud"
EMBEDDING_MODEL    = "nomic-embed-text"
STRENGTH_MODEL     = "minimax-m2.7:cloud"
GENERATION_MODEL   = "minimax-m2.7:cloud"
