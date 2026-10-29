import os
import sys
import time
import requests
import httpx
from google import genai
from google.genai import types

# Load Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_message(message: str) -> bool:
    """Sends generated report text to the specified Telegram Chat."""
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
            print(f"Telegram API returned error: {result}")
            return False

        print("Successfully sent stock report to Telegram!")
        return True

    except requests.exceptions.Timeout:
        print("Telegram request timed out.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Telegram request failed: {e}")
        return False


def generate_market_report(client: genai.Client, model_name: str, prompt: str) -> str:
    """
    Streams response from Gemini to keep connection active and prevent 
    [Errno 104] socket connection resets.
    """
    response_stream = client.models.generate_content_stream(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
        )
    )

    full_text = []
    for chunk in response_stream:
        if chunk.text:
            full_text.append(chunk.text)

    return "".join(full_text)


def main():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing from environment variables/secrets.")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Telegram credentials missing from environment variables/secrets.")

    # Configure custom HTTP client with extended timeouts
    custom_httpx_client = httpx.Client(
        timeout=httpx.Timeout(connect=60.0, read=600.0, write=60.0, pool=60.0),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    )

    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={"httpx_client": custom_httpx_client}
    )

    # Updated active model list
    models_to_try = ['gemini-3.8-flash', 'gemini-3.5-flash-lite']
    prompt = 'Provide a concise daily stock market overview and key trends to watch today.'

    for model_name in models_to_try:
        for attempt in range(1, 6):
            try:
                print(f"Generating content using model: {model_name} (Attempt {attempt})...")
                
                report_text = generate_market_report(client, model_name, prompt)

                if report_text:
                    print(f"Successfully generated content with {model_name}!")
                    sent = send_telegram_message(report_text)
                    if sent:
                        return
                    else:
                        raise RuntimeError("Failed to deliver message via Telegram.")

            except Exception as e:
                print(f"Attempt {attempt} for model {model_name} failed: {type(e).__name__} - {e}")
                if attempt < 5:
                    wait_time = attempt * 10
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)

    raise RuntimeError("All configured model attempts and retries failed.")


if __name__ == "__main__":
    main()
