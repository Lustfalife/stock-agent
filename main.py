import os
import time
import requests
from google import genai
from google.genai.errors import ServerError, APIError

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        # Connect timeout: 10s, Read timeout: 60s
        response = requests.post(url, json=payload, timeout=(10, 60))
        response.raise_for_status()

        result = response.json()
        if not result.get("ok"):
            print(f"Telegram API returned failure: {result}")
            return False

        print("Successfully sent stock report to Telegram!")
        return True

    except requests.exceptions.Timeout:
        print("Telegram request timed out.")
        return False

    except requests.exceptions.RequestException as e:
        print(f"Telegram request failed: {e}")
        return False

def main():
    # Sanity check secrets
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing from environment variables/secrets.")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Telegram token or Chat ID is missing from environment variables/secrets.")

    # Initialize Gemini client
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Try current standard models
    models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash']
    
    for model_name in models_to_try:
        try:
            print(f"Generating content with model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents='Provide a concise daily stock market overview and key trends to watch today.',
            )
            
            if response and response.text:
                print("Successfully generated content from Gemini!")
                sent = send_telegram_message(response.text)
                if sent:
                    return
                else:
                    raise RuntimeError("Failed to deliver message via Telegram.")
            
        except (ServerError, APIError) as e:
            print(f"Model {model_name} failed with API error: {e}")
            time.sleep(3)

    raise RuntimeError("All model attempts failed.")

if __name__ == "__main__":
    main()
