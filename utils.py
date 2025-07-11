# utils.py

import os
import yaml
import requests
import streamlit as st

def load_users():
    try:
        if not os.path.exists("users.yaml"):
            with open("users.yaml", "w") as f:
                yaml.dump({"credentials": {"usernames": {}}}, f)
            return {"credentials": {"usernames": {}}}

        with open("users.yaml", "r") as f:
            data = yaml.safe_load(f)
            if data is None:
                return {"credentials": {"usernames": {}}}
            if "credentials" not in data or "usernames" not in data["credentials"]:
                data["credentials"] = {"usernames": {}}
            return data
    except Exception as e:
        st.error(f"Eroare la încărcarea utilizatorilor: {e}")
        return {"credentials": {"usernames": {}}}

def send_telegram_message(message, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=payload)
    return response.ok
