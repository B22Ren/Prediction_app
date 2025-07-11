from datetime import date, timedelta
from predictii_joc import actualizeaza_valoare_reală, calculeaza_scoruri
import requests
import yaml
import os


ieri = date.today() - timedelta(days=1)
ieri_str = ieri.isoformat()

if not os.path.exists("joc_predictii.yaml"):
    print("Fișierul joc_predictii.yaml nu există.")
    exit()

with open("joc_predictii.yaml", "r") as f:
    data = yaml.safe_load(f) or []

valute_de_actualizat = set(
    entry["valuta"]
    for entry in data
    if entry["data"] == ieri_str and entry["valoare_real"] is None
)

if not valute_de_actualizat:
    print("📭 Nicio valoare reală de completat pentru ieri.")
else:
    print(f"📈 Valute de completat pentru {ieri_str}: {', '.join(valute_de_actualizat)}")


    try:
        url = f"https://api.frankfurter.app/{ieri_str}?from=USD"
        response = requests.get(url)
        response.raise_for_status()
        rates = response.json()["rates"]

        for valuta in valute_de_actualizat:
            valoare = rates.get(valuta.split("/")[-1])
            if valoare:
                success = actualizeaza_valoare_reală(ieri_str, valuta, valoare)
                if success:
                    print(f"Valoare reală setată pentru {valuta} = {valoare}")
            else:
                print(f"Nu există curs pentru {valuta} în API.")

    except Exception as e:
        print(f" Eroare la obținerea datelor valutare: {e}")
        exit()

if calculeaza_scoruri():
    print("🎯 Scoruri actualizate.")
else:
    print(" Nicio actualizare de scor necesară.")
