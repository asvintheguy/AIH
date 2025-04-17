"""
Configuration settings for the Health Risk Assessment Chatbot
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API key is now handled by the environment variable OPENAI_API_KEY
# OpenAI base URL is now handled by the environment variable OPENAI_BASE_URL

# Model settings
DEFAULT_MODEL = "gpt-4"
TEMPERATURE = 0.3
MAX_TOKENS = 150

# Dataset settings
DOWNLOAD_PATH = "kaggle_dataset"

# Model training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42 