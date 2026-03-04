"""Configuration related to the Amplify API and the server setup."""
import os
import dotenv

dotenv.load_dotenv()

AMPLIFY_BASE_URL = "https://prod-api.vanderbilt.ai"

# Use the environment variable, but define a fallback helper if needed.
# Auth logic will import from here or directly check `os.getenv`
