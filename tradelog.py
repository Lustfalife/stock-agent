"""
tradelog.py - sends trades to your Google Sheet ("Trade Log") and reads them back.
Uses two GitHub secrets: TRADELOG_URL and TRADELOG_KEY.
If the sheet is unreachable, it only prints a message and never stops the scanner.
"""
import os
import json
import requests


def _post(payload):
    url = os.getenv("TRADELOG_URL")
    key = os.getenv("TRADELOG_KEY")
    if not url or not key:
        print("Trade log not configured (TRADELOG_URL / TRADELOG_KEY missing).")
        return None
    try:
        body = dict(payload, key=key)
        r = requests.post(url, data=json.dumps(body),
                          headers={"Content-Type": "text/plain"}, timeout=60)
        res = r.json()
        if not res.get("ok"):
            print("Trade log error:", res.get("error"))
            return None
        return res
    except Exception as e:
        print("Trade log request failed:", e)
        return None


def add_rows(tab, rows):
    return _post({"action": "add", "tab": tab, "rows": rows})


def open_rows(tab):
    res = _post({"action": "open", "tab": tab})
    return res["rows"] if res else None


def update_rows(tab, updates):
    return _post({"action": "update", "tab": tab, "updates": updates})
