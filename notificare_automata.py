import pandas as pd
import yaml
import os
from datetime import datetime
from rnn import rnn_model
from arima import build_model_predict_arima, forecast_future_days
from utils import send_telegram_message, load_users


def trimite_notificari_pentru_toti_utilizatorii():
    if not os.path.exists("notificari.yaml") or not os.path.exists("currency_data.csv"):
        print("⚠️ Lipsesc fișierele necesare.")
        return

    with open("notificari.yaml", "r") as f:
        setari = yaml.safe_load(f) or {}

    users_data = load_users()
    df = pd.read_csv("currency_data.csv", index_col=0)
    df.index = pd.to_datetime(df.index)

    for utilizator, info in setari.items():
        # ⚠️ Verificăm dacă info e dict valid
        if not isinstance(info, dict):
            print(f"⚠️ Setările pentru {utilizator} sunt invalide. Ignorat.")
            continue

        valuta = info.get("valuta")
        model = info.get("model")
        chat_id = users_data["credentials"]["usernames"].get(utilizator, {}).get("telegram_chat_id", "")

        if not chat_id:
            print(f"⚠️ {utilizator} nu are Telegram Chat ID configurat.")
            continue

        try:
            valori = df[valuta].dropna()
            if valori.empty:
                print(f"⚠️ Date lipsă pentru {valuta}.")
                continue

            if model == "ARIMA":
                model_fit, _, pred_train = build_model_predict_arima(valori.tolist(), [])
                forecast = forecast_future_days(model_fit, pred_train, 1)
                valoare = forecast[0]
            elif model == "RNN":
                result = rnn_model(valuta, 1)
                valoare = result["future_predictions"][0]
            else:
                print(f"⚠️ Model invalid pentru {utilizator}: {model}")
                continue

            mesaj = (
                f"🔔 Predicția pentru {valuta} ({model}) la data de "
                f"{datetime.now().date() + pd.Timedelta(days=1)} este: {valoare:.4f}"
            )

            success = send_telegram_message(mesaj, os.getenv("TELEGRAM_TOKEN"), chat_id)
            if success:
                print(f"✅ Mesaj trimis către {utilizator}")
            else:
                print(f"❌ Eroare la trimitere pentru {utilizator}")

        except Exception as e:
            print(f"❌ Eroare pentru utilizatorul {utilizator}: {e}")
