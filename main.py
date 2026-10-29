
import os
import time
import requests
from google import genai
from google.genai.errors import ServerError

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
        timeout=30,
    )
    response.raise_for_status()


def main():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing from GitHub Secrets.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents="Provide a concise daily stock market overview and key trends to watch today.",
            )

            if not response.text:
                raise ValueError("Gemini returned an empty report.")

            send_telegram_message(response.text)
            print("Successfully sent stock report to Telegram!")
            return

        except ServerError as e:
            print(f"Temporary Gemini server error: {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                raise

if __name__ == "__main__":
    main()
