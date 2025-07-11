import yaml
import os
import pandas as pd
from datetime import date
from collections import defaultdict
from arima import build_model_predict_arima, forecast_future_days
from rnn import rnn_model

def adauga_predictie_user(username, valuta, predictie_user, model_ai="ARIMA"):
    azi = str(date.today())
    if os.path.exists("joc_predictii.yaml"):
        with open("joc_predictii.yaml", "r") as f:
            data = yaml.safe_load(f) or []
    else:
        data = []

    for entry in data:
        if entry["username"] == username and entry["valuta"] == valuta and entry["data"] == azi:
            return False

    valoare_ai = genereaza_predictie_ai(valuta, model_ai)

    intrare = {
        "username": username,
        "data": azi,
        "valuta": valuta,
        "predictie_user": float(predictie_user),
        "predictie_ai": float(valoare_ai),
        "valoare_real": None,
        "scor_user": None,
        "scor_ai": None
    }

    data.append(intrare)
    with open("joc_predictii.yaml", "w") as f:
        yaml.dump(data, f)

    return True

def genereaza_predictie_ai(valuta, model_ai="ARIMA"):
    df = pd.read_csv("currency_data.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    serie = df[valuta].dropna()

    if model_ai == "ARIMA":
        model_fit, _, pred_train = build_model_predict_arima(serie.tolist(), [])
        predictie = forecast_future_days(model_fit, pred_train, 1)
        return round(predictie[0], 4)

    elif model_ai == "RNN":
        rezultat = rnn_model(valuta, 1)
        return round(rezultat["future_predictions"][0], 4)

    else:
        raise ValueError("Model necunoscut: trebuie să fie ARIMA sau RNN")

def actualizeaza_valoare_reală(data_str, valuta, valoare_reală):
    if not os.path.exists("joc_predictii.yaml"):
        return False

    with open("joc_predictii.yaml", "r") as f:
        data = yaml.safe_load(f) or []

    modificat = False
    for entry in data:
        if entry["data"] == data_str and entry["valuta"] == valuta:
            entry["valoare_real"] = float(valoare_reală)
            modificat = True

    if modificat:
        with open("joc_predictii.yaml", "w") as f:
            yaml.dump(data, f)

    return modificat

def calculeaza_scoruri():
    if not os.path.exists("joc_predictii.yaml"):
        return False

    with open("joc_predictii.yaml", "r") as f:
        data = yaml.safe_load(f) or []

    actualizat = False
    for entry in data:
        if entry["valoare_real"] is not None and entry["scor_user"] is None:
            vr = float(entry["valoare_real"])
            pu = float(entry["predictie_user"])
            pa = float(entry["predictie_ai"])

            entry["scor_user"] = max(0, round(100 - abs(pu - vr) * 1000))
            entry["scor_ai"] = max(0, round(100 - abs(pa - vr) * 1000))
            actualizat = True

    if actualizat:
        with open("joc_predictii.yaml", "w") as f:
            yaml.dump(data, f)

    return actualizat

def genereaza_clasament():
    if not os.path.exists("joc_predictii.yaml"):
        return []

    with open("joc_predictii.yaml", "r") as f:
        data = yaml.safe_load(f) or []

    scoruri = defaultdict(int)
    for entry in data:
        if entry["scor_user"] is not None:
            scoruri[entry["username"]] += entry["scor_user"]

    clasament = sorted(scoruri.items(), key=lambda x: x[1], reverse=True)
    return clasament

def insigne_utilizator(username):
    if not os.path.exists("joc_predictii.yaml"):
        return []

    with open("joc_predictii.yaml", "r") as f:
        data = yaml.safe_load(f) or []

    insigne = set()
    consecutive_high = 0

    for entry in sorted(data, key=lambda x: x["data"]):
        if entry["username"] != username or entry["scor_user"] is None:
            continue

        if entry["scor_user"] > entry["scor_ai"]:
            insigne.add("🥇 Ai învins AI-ul")

        if abs(entry["predictie_user"] - entry["valoare_real"]) < 0.01:
            insigne.add("🎯 Sniper")

        if entry["scor_user"] >= 80:
            consecutive_high += 1
            if consecutive_high >= 5:
                insigne.add("🔥 5 zile cu scor >80")
        else:
            consecutive_high = 0

    return list(insigne)
