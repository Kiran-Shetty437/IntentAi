import requests
import os

class IntentAI:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API key is required")

        self.api_key = api_key

        # 🔒 URL is hidden here (like OpenAI SDK)
        self.base_url = os.getenv(
            "INTENT_AI_URL",
            "https://intentai-pqtt.onrender.com"
        ).rstrip("/")

    def classify(self, text):
        headers = {
            "x-api-key": self.api_key
        }
        payload = {
            "text": text
        }

        response = requests.post(
            f"{self.base_url}/intent",
            json=payload,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()
        return response.json()
