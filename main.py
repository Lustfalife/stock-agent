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
        response = requests.post(url, json=payload, timeout=(10, 60))
        response.raise_for_status()

        result = response.json()
        if not result.get("ok"):
            raise RuntimeError(f"Telegram API error: {result}")

        print("Successfully sent stock report to Telegram!")
        return True

    except requests.exceptions.Timeout:
        print("Telegram timed out. Delivery status is unknown; not retrying automatically.")
        return False

    except requests.exceptions.RequestException as e:
        print(f"Telegram request failed: {e}")
        return False

def main():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing from GitHub Secrets.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Try gemini-3.8-flash first as instructed by the API error logs, falling back to gemini-2.0-flash
    models_to_try = ['gemini-3.8-flash', 'gemini-2.0-flash']
    
    for model_name in models_to_try:
        try:
            print(f"Generating content using model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents='Provide a concise daily stock market overview and key trends to watch today.',
            )
            if response.text:
                send_telegram_message(response.text)
                return
        except (ServerError, APIError) as e:
            print(f"Failed with {model_name}: {e}")
            time.sleep(2)

    raise RuntimeError("All configured model attempts failed.")

if __name__ == "__main__":
    main()
