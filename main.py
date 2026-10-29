import os
import time
import requests
from google import genai
from google.genai.errors import ServerError

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    response = requests.post(url, json=payload)
    return response.json()

def main():
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Retry loop in case Google API experiences high demand (503)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents='Provide a concise daily stock market overview and key trends to watch today.',
            )
            send_telegram_message(response.text)
            print("Successfully sent stock report to Telegram!")
            break
        except ServerError as e:
            print(f"Attempt {attempt + 1} hit temporary server demand. Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()
