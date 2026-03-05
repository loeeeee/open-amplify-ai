import requests
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("AMPLIFY_AI_TOKEN")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

data = {
    "data": {
        "temperature": 0.0,
        "max_tokens": 4000,
        "dataSources": [],
        "messages": [
            {"role": "user", "content": "Run a tool"},
            {"role": "assistant", "content": "{\"command\": \"list_files\", \"parameters\": {}}"},
            {"role": "tool", "content": "file1.txt"}
        ],
        "options": {
            "model": {"id": "o4-mini"},
        },
    }
}

try:
    resp = requests.post("https://prod-api.vanderbilt.ai/chat", headers=headers, json=data)
    print("Status tool role:", resp.status_code)
    print("Response text tool role:", resp.text)
except Exception as e:
    print("Error:", e)
