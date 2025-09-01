import os
from pathlib import Path

# Max number of messages to keep in memory
MAX_HISTORY = 50

# Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "chatbot_log.json"

# GPT2 Hyperparameters
GPT2_MODEL_NAME = "gpt2-medium"
MAX_TOKENS = 1024
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9

# Input processor configs
REMOVE_PUNCTUATION = True
LOWERCASE = True

# Post-processor configs
CENSOR_WORDS = ["idiot", "stupid", "dumb"]
TRIM_WHITESPACE = True

# Misc
DEBUG_MODE = True
ENGINE_TYPE = "GPT2"
