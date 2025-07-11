import streamlit as st
import yaml
import bcrypt
import os
from os.path import exists
import yagmail
from dotenv import load_dotenv
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import timedelta
import statsmodels.api as sm
from arima import load_data_set, build_model_predict_arima, forecast_future_days, evaluate_performance_arima, plot_arima
from rnn import rnn_model
from datetime import datetime
import requests
from collections import defaultdict
from predictii_joc import (
    adauga_predictie_user,
    genereaza_clasament,
    insigne_utilizator,
    actualizeaza_valoare_reală,
    calculeaza_scoruri
)
from predictii_joc import actualizeaza_valoare_reală, calculeaza_scoruri
from notificare_automata import trimite_notificari_pentru_toti_utilizatorii


load_dotenv()
EMAIL = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")


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
        st.error(f"Error loading user data: {e}")
        return {"credentials": {"usernames": {}}}


def send_telegram_message(message, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=payload)
    return response.ok

def save_users(users):
    try:
        with open("users.yaml", "w") as f:
            yaml.dump(users, f, default_flow_style=False)
        st.success("User data saved successfully.")
    except Exception as e:
        st.error(f"Error saving user data: {e}")

def salveaza_in_istoric(model, valuta, zile, predictii, utilizator):
    intrare = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "valuta": valuta,
        "zile": zile,
        "utilizator": utilizator,
        "predictii": [float(v) for v in predictii]
    }



    if not os.path.exists("istoric.yaml"):
        with open("istoric.yaml", "w") as f:
            yaml.dump({"predictii": [intrare]}, f)
    else:
        with open("istoric.yaml", "r") as f:
            data = yaml.safe_load(f) or {"predictii": []}
        data["predictii"].append(intrare)
        with open("istoric.yaml", "w") as f:
            yaml.dump(data, f)

def sterge_istoric_utilizator(username):
    try:
        if not os.path.exists("istoric.yaml"):
            return False

        with open("istoric.yaml", "r") as f:
            data = yaml.safe_load(f) or {"predictii": []}

        vechi = data.get("predictii", [])
        noi = [p for p in vechi if p.get("utilizator") != username]

        print(f"[DEBUG] Total inițial: {len(vechi)}, după filtrare: {len(noi)}")

        with open("istoric.yaml", "w") as f:
            yaml.dump({"predictii": noi}, f)

        return True
    except Exception as e:
        print(f"⚠️ Eroare la ștergere: {e}")
        return False



# --- Session state ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Acasă"
if "username" not in st.session_state:
    st.session_state["username"] = None



st.set_page_config(page_title="Autentificare și Navigare", layout="centered")
st.title("R€-Course")

# --- Auth ---
if not st.session_state["logged_in"]:
    users_data = load_users()
    menu = st.sidebar.selectbox("Alege acțiunea:", ["Login", "Register", "Forgot Password"], key="auth_menu")

    if menu == "Register":
        st.subheader("🆕 Înregistrare")
        new_username = st.text_input("Nume utilizator")
        new_email = st.text_input("Email")
        new_password = st.text_input("Parolă", type="password")
        new_telegram_id = st.text_input("Telegram Chat ID (opțional)")
       


        if st.button("Creează cont"):
            if not new_username or not new_email or not new_password:
                st.error("Toate câmpurile sunt obligatorii.")
            elif new_username in users_data["credentials"]["usernames"]:
                st.error("Utilizatorul există deja.")
            else:
                hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                users_data["credentials"]["usernames"][new_username] = {
                    "email": new_email,
                    "name": new_username,
                    "password": hashed_pw,
                    "telegram_chat_id": new_telegram_id.strip() if new_telegram_id else ""
                }
                save_users(users_data)
                st.success("Cont creat cu succes! Te poți autentifica acum.")

    elif menu == "Login":
        st.subheader("🔑 Autentificare")
        username_input = st.text_input("Nume utilizator")
        password_input = st.text_input("Parolă", type="password")

        if st.button("Autentifică-te"):
            if username_input in users_data["credentials"]["usernames"]:
                hashed_pw = users_data["credentials"]["usernames"][username_input]["password"]
                if bcrypt.checkpw(password_input.encode(), hashed_pw.encode()):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username_input
                    st.success(f"Bine ai venit, {username_input}!")
                    st.rerun()
                else:
                    st.error("Parolă greșită.")
            else:
                st.error("Utilizatorul nu există.")

    elif menu == "Forgot Password":
        st.subheader("🔁 Recuperare parolă")
        email_input = st.text_input("Emailul asociat contului")

        if st.button("Trimite email de resetare"):
            if not email_input:
                st.warning("Introdu adresa de email.")
            else:
                users_found = [
                    (user, info)
                    for user, info in users_data["credentials"]["usernames"].items()
                    if info.get("email") == email_input
                ]
                if users_found:
                    user, info = users_found[0]
                    if EMAIL and EMAIL_PASSWORD:
                        try:
                            yag = yagmail.SMTP(user=EMAIL, password=EMAIL_PASSWORD)
                            yag.send(
                                to=email_input,
                                subject="Recuperare parolă",
                                contents=f"Salut {info['name']}, contactează administratorul pentru resetare."
                            )
                            st.success("Email trimis cu succes.")
                        except Exception as e:
                            st.error(f"Eroare la trimiterea emailului: {e}")
                    else:
                        st.error("Credențialele de email lipsesc din .env.")
                else:
                    st.error("Emailul nu a fost găsit în baza de date.")
    st.stop() 



# --- Dashboard ---
else:
    if st.session_state["username"] is None:
        st.error("Eroare: utilizatorul nu este autentificat corect.")
        st.session_state["logged_in"] = False
        st.rerun()

    users_data = load_users()
    st.sidebar.title("Navigare")
    user_info = users_data["credentials"]["usernames"].get(st.session_state["username"], {})
    profile_img = user_info.get("profile_image")
    if profile_img and os.path.exists(profile_img):
        st.sidebar.image(profile_img, width=100)
    st.sidebar.markdown(f"**{st.session_state['username']}**")

    page = st.sidebar.radio("Alege pagina:", ["Acasă", "Predicție ARIMA", "Predicție RNN", "Setări Notificări", "Setări Cont","Istoric Predicții","Joc de predicții", "Delogare"])
    st.session_state["current_page"] = page


if page == "Acasă":
    st.header("🏠 Bine ai venit în aplicația de predicție valutară!")
    st.write("Această aplicație îți permite să consulți cursuri valutare live, să efectuezi conversii valutare și să realizezi predicții folosind modele ARIMA și RNN.")

    st.subheader("🌐 Cursuri valutare live")
    try:
        response = requests.get("https://api.frankfurter.app/latest")
        if response.status_code == 200:
            data = response.json()
            base = data["base"]
            rates = data["rates"]
            date = data["date"]

            df_rates = pd.DataFrame(rates.items(), columns=["Valută", f"Curs față de {base}"])
            st.write(f"Cursuri valutare la data de {date} (bază: {base})")
            st.dataframe(df_rates.sort_values("Valută").reset_index(drop=True))

            st.subheader("💱 Conversie valutară")
            from_currency = st.selectbox("Convertesc din:", [base] + list(rates.keys()))
            to_currency = st.selectbox("În:", list(rates.keys()))
            amount = st.number_input("Suma de convertit", min_value=0.0, value=1.0, step=0.1)

            if from_currency == base:
                converted = amount * rates[to_currency]
            elif to_currency == base:
                converted = amount / rates[from_currency]
            else:
                eur_amount = amount / rates[from_currency]
                converted = eur_amount * rates[to_currency]

            st.success(f"{amount:.2f} {from_currency} = {converted:.2f} {to_currency}")

            st.subheader("📈 Vizualizare grafică modernă")

           
            df_rates_sorted = df_rates.sort_values("Valută")
            valute = df_rates_sorted["Valută"]
            valori = df_rates_sorted[f"Curs față de {base}"]

            
            threshold = valori.quantile(0.95)
            valute_filtrate = valute[valori < threshold]
            valori_filtrate = valori[valori < threshold]

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(valute_filtrate, valori_filtrate, color="#1f77b4")

            ax.set_ylabel("Curs")
            ax.set_title(f"Cursuri valutare ({date})")
            ax.set_xticks(range(len(valute_filtrate)))
            ax.set_xticklabels(valute_filtrate, rotation=45, ha="right", fontsize=9)
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            st.pyplot(fig)
        else:
            st.error("Nu s-au putut obține datele live de la API-ul Frankfurter.")
    except Exception as e:
        st.error(f"Eroare la încărcarea cursurilor valutare: {e}")

elif page == "Setări Cont":
    user_data = users_data["credentials"]["usernames"][st.session_state["username"]]

    def card_container(title, content_function):
        st.markdown(
            f"""
            <div style='background-color:#1e1e2f;padding:20px;border-radius:12px;margin:15px 0;border:1px solid #444'>
                <h4 style='color:#fff;'>{title}</h4>
            """,
            unsafe_allow_html=True
        )
        content_function()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h2 style='color:#fff;'>👤 Setările contului</h2>", unsafe_allow_html=True)

    def email_card():
        new_email = st.text_input("Email nou", value=user_data.get("email", ""), label_visibility="collapsed")
        if st.button("💾 Salvează emailul"):
            user_data["email"] = new_email.strip()
            save_users(users_data)
            st.success("📧 Email actualizat!")

    card_container("📧 Schimbă adresa de email", email_card)

    def password_card():
        current_password = st.text_input("Parolă actuală", type="password", label_visibility="collapsed")
        new_password = st.text_input("Parolă nouă", type="password", label_visibility="collapsed")
        if st.button("🔒 Actualizează parola"):
            if bcrypt.checkpw(current_password.encode(), user_data["password"].encode()):
                user_data["password"] = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                save_users(users_data)
                st.success("✅ Parola a fost schimbată.")
            else:
                st.error("❌ Parola actuală este greșită.")

    card_container("🔐 Schimbă parola", password_card)

    def telegram_card():
        new_chat_id = st.text_input("Telegram Chat ID", value=user_data.get("telegram_chat_id", ""), label_visibility="collapsed")
        if st.button("💬 Salvează Telegram ID"):
            user_data["telegram_chat_id"] = new_chat_id.strip()
            save_users(users_data)
            st.success("💬 Telegram Chat ID actualizat.")

    card_container("💬 Telegram Chat ID", telegram_card)

    def notificari_card():
        enable_notifications = st.checkbox("Activează notificările zilnice", value=user_data.get("notifications_enabled", False))
        if st.button("💾 Salvează notificările"):
            user_data["notifications_enabled"] = enable_notifications
            save_users(users_data)
            st.success("🔔 Setările de notificare au fost salvate.")

    card_container("🔔 Notificări zilnice", notificari_card)

    def imagine_card():
        uploaded_img = st.file_uploader("Încarcă imagine", type=["png", "jpg", "jpeg"])
        if uploaded_img:
            img_path = f"profile_images/{st.session_state['username']}.png"
            os.makedirs("profile_images", exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(uploaded_img.read())
            st.image(img_path, caption="Imagine actualizată", width=150)
            user_data["profile_image"] = img_path
            save_users(users_data)
            st.success("🖼️ Imagine de profil actualizată.")

    card_container("🖼️ Imagine de profil", imagine_card)


elif page == "Predicție ARIMA":
    st.header("📈 Predicție ARIMA")
    uploaded_file = st.file_uploader("Încarcă fișierul CSV", type="csv")

    if uploaded_file:
        with open("currency_data.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())

        df = pd.read_csv("currency_data.csv")

        if df.empty or df.shape[0] < 2:
            st.error("Fișierul trebuie să conțină cel puțin două rânduri de date.")
            st.stop()

        date_col = None
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                date_col = col
                break
            except:
                continue

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        else:
            st.error("Fișierul nu conține o coloană de tip dată.")
            st.stop()

        st.write("Date încărcate:", df.head())

        currency_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if not currency_columns:
            st.error("Fișierul nu conține coloane numerice (valute).")
        else:
            valuta = st.selectbox("Alege valuta:", currency_columns, key="valuta_arima")
            prediction_mode = st.radio("Alege modul de predicție:", ["Pe număr de zile", "Pe dată exactă"], key="mode_arima")

            if prediction_mode == "Pe număr de zile":
                days = st.slider("Număr de zile de prezis:", 1, 30, 7)
            else:
                last_date = df.index[-1]
                selected_date = st.date_input("Selectează data pentru predicție", min_value=last_date + timedelta(days=1))
                days = (selected_date - last_date.date()).days
                if days <= 0:
                    st.warning("Te rog alege o dată după ultima dată din fișier.")
                    st.stop()

            if st.button("Execută predicția ARIMA"):
                try:
                    raw_data = df[valuta].dropna().tolist()
                    dates = df.index.tolist()
                    training, testing = raw_data[:-days], raw_data[-days:]
                    model_fit, pred_test, pred_train = build_model_predict_arima(training, testing)
                    forecast = forecast_future_days(model_fit, pred_train, days)

                    result_df = pd.DataFrame({
                        "Data": pd.date_range(start=dates[-1] + timedelta(days=1), periods=days),
                        "Valoare prezisă": forecast
                    })

                    st.dataframe(result_df)

                    
                    salveaza_in_istoric(
                        model="ARIMA",
                        valuta=valuta,
                        zile=days,
                        predictii=forecast,
                        utilizator=st.session_state.get("username", "necunoscut")
                    )

                    plot_arima(valuta, testing, pred_test, "predictie_arima.pdf", training_series=training)

                    with open("predictie_arima.pdf", "rb") as f:
                        st.download_button("Descarcă graficul PDF", f, "predictie_arima.pdf")

                except Exception as e:
                    st.error(f"Eroare: {e}")

elif page == "Predicție RNN":
    st.header("📉 Predicție RNN")
    uploaded_file = st.file_uploader("Încarcă fișierul CSV", type="csv", key="rnn_csv")

    if uploaded_file:
        try:
            with open("currency_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())

            df = pd.read_csv("currency_data.csv", index_col=0)

            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                st.error(f"Indexul nu este de tip dată: {e}")
                st.stop()

            if df.empty or df.shape[0] < 2:
                st.error("Fișierul trebuie să conțină cel puțin două rânduri de date.")
                st.stop()

            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            if not numeric_cols:
                st.error("Fișierul nu conține coloane numerice pentru predicție.")
                st.stop()

            st.write("Date încărcate:", df.head())

            valuta = st.selectbox("Alege valuta:", numeric_cols, key="valuta_rnn")
            prediction_mode = st.radio("Alege modul de predicție:", ["Pe număr de zile", "Pe dată exactă"], key="mode_rnn")

            if prediction_mode == "Pe număr de zile":
                days = st.slider("Număr de zile de prezis:", 1, 30, 10)
            else:
                last_date = df.index[-1]
                selected_date = st.date_input("Selectează data pentru predicție (RNN)", min_value=last_date + timedelta(days=1))
                days = (selected_date - last_date.date()).days
                if days <= 0:
                    st.warning("Te rog alege o dată după ultima dată din fișier.")
                    st.stop()

            if st.button("Execută predicția RNN"):
                try:
                    result = rnn_model(valuta, days)

                    if not result or "future_predictions" not in result:
                        st.error("Rezultatul modelului RNN este invalid sau incomplet.")
                        st.stop()

                    salveaza_in_istoric(
                        model="RNN",
                        valuta=valuta,
                        zile=days,
                        predictii=result["future_predictions"],
                        utilizator=st.session_state.get("username", "necunoscut")
                    )

                    result_df = pd.DataFrame({
                        "Ziua": [f"Ziua {i+1}" for i in range(days)],
                        "Valoare prezisă": result["future_predictions"]
                    })

                    st.dataframe(result_df)

                    with open("predictie_rnn.pdf", "rb") as f:
                        st.download_button("Descarcă graficul PDF", f, "predictie_rnn.pdf")

                except Exception as e:
                    st.error(f"Eroare în execuția modelului RNN: {e}")

        except Exception as e:
            st.error(f"Eroare la procesarea fișierului: {e}")
elif page == "Setări Notificări":
    st.header("🔔 Setări notificări zilnice")
    st.info("Configurează notificările pentru a primi predicții zilnice prin Telegram.")

    users_data = load_users()
    utilizator = st.session_state["username"]
    user_info = users_data["credentials"]["usernames"][utilizator]
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

    if st.button("🔔 Trimite notificare test", key="btn_test_telegram"):
        TELEGRAM_CHAT_ID = user_info.get("telegram_chat_id", "")
        mesaj_test = "🔔 Acesta este un test de notificare din aplicația de predicții!"

        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            sent = send_telegram_message(mesaj_test, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
            if sent:
                st.success("✅ Test trimis cu succes în Telegram!")
            else:
                st.warning("⚠️ Eroare la trimiterea testului.")
        else:
            st.warning("⚠️ ID-ul de Telegram nu este configurat în contul tău.")

    if os.path.exists("currency_data.csv"):
        df = pd.read_csv("currency_data.csv", index_col=0)
        try:
            df.index = pd.to_datetime(df.index)
            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

            if not numeric_cols:
                st.error("Fișierul valutar nu conține coloane numerice.")
            else:
                valuta_nt = st.selectbox("Alege valuta pentru notificare:", numeric_cols, key="valuta_notify")
                model_nt = st.selectbox("Alege modelul de predicție:", ["ARIMA", "RNN"], key="model_notify")

                if st.button("💾 Salvează setările de notificare", key="btn_save_notify"):
                    try:
                        email_utilizator = user_info["email"]

                        # Încarcă toate setările existente
                        notificari = {}
                        if os.path.exists("notificari.yaml"):
                            with open("notificari.yaml", "r") as f:
                                notificari = yaml.safe_load(f) or {}

                        # Actualizează doar datele utilizatorului curent
                        notificari[utilizator] = {
                            "valuta": valuta_nt,
                            "model": model_nt,
                            "email": email_utilizator
                        }

                        with open("notificari.yaml", "w") as f:
                            yaml.dump(notificari, f)

                        st.success("✅ Setările au fost salvate. Vei primi zilnic un mesaj cu predicția.")
                    except Exception as e:
                        st.error(f"Eroare la salvarea setărilor: {e}")

                if st.button("📤 Cere predicția pentru mâine", key="btn_request_prediction"):
                    try:
                        df_valuta = df[valuta_nt].dropna()
                        if df_valuta.empty:
                            st.warning("Datele pentru valuta selectată sunt goale.")
                            st.stop()

                        if model_nt == "ARIMA":
                            training = df_valuta.tolist()
                            model_fit, _, pred_train = build_model_predict_arima(training, [])
                            forecast = forecast_future_days(model_fit, pred_train, 1)
                            valoare = forecast[0]
                            mesaj_pred = f"📈 Predicția pentru mâine ({valuta_nt}, ARIMA): {valoare:.4f}"
                        else:
                            result = rnn_model(valuta_nt, 1)
                            valoare = result["future_predictions"][0]
                            mesaj_pred = f"📉 Predicția pentru mâine ({valuta_nt}, RNN): {valoare:.4f}"

                        st.success(mesaj_pred)

                        TELEGRAM_CHAT_ID = user_info.get("telegram_chat_id", "")
                        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                            sent = send_telegram_message(mesaj_pred, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
                            if sent:
                                st.success("✅ Predicția a fost trimisă și pe Telegram.")
                            else:
                                st.warning("⚠️ Nu s-a putut trimite mesajul pe Telegram.")
                        else:
                            st.warning("⚠️ ID-ul de Telegram nu este configurat în contul tău.")
                    except Exception as e:
                        st.error(f"Eroare la generarea predicției: {e}")

        except Exception as e:
            st.error(f"Eroare la procesarea fișierului: {e}")
    else:
        st.warning("Nu a fost găsit niciun fișier de date valutar. Încarcă unul în pagina ARIMA sau RNN.")

    # 🔘 Buton nou: Trimite notificări la toți utilizatorii
    if st.button("📤 Trimite notificări tuturor utilizatorilor", key="btn_notify_all"):
        trimite_notificari_pentru_toti_utilizatorii()
        st.success("✅ Notificările au fost trimise către toți utilizatorii.")


#elif page == "Istoric Predicții":
 #   st.write("👤 Utilizator activ:", st.session_state.get("username"))

  #  st.header("🗂️ Istoric Predicții")
  #  if os.path.exists("istoric.yaml"):
  #      with open("istoric.yaml", "r") as f:
  #          data = yaml.safe_load(f)
   #         for intrare in reversed(data["predictii"]):
   #             with st.expander(f"{intrare['timestamp']} • {intrare['model']} • {intrare['valuta']}"):
   #                 st.write(f"Utilizator: {intrare['utilizator']}")
   #                 st.write(f"Zile prezise: {intrare['zile']}")
   #                 st.write("Valori prezise:", intrare["predictii"])
   # else:
    #    st.info("Nu există predicții salvate încă.")


elif page == "Istoric Predicții":
    st.header("📜 Istoric Predicții")

    if not os.path.exists("istoric.yaml"):
        st.info("Nu există predicții salvate încă.")
    else:
        try:
            with open("istoric.yaml", "r") as f:
                data = yaml.safe_load(f) or {}

            toate_pred = data.get("predictii", [])
            user_pred = [p for p in toate_pred if p.get("utilizator") == st.session_state["username"]]

            if not user_pred:
                st.info("Nu ai încă predicții salvate.")
            else:
                st.markdown("### 📂 Predicțiile tale salvate")
                for entry in reversed(user_pred):
                    with st.expander(f"{entry['timestamp']} • {entry['model']} • {entry['valuta']}"):
                        st.markdown(f"""
                        - **Model:** {entry['model']}
                        - **Valută:** {entry['valuta']}
                        - **Zile prezise:** {entry['zile']}
                        - **Timestamp:** {entry['timestamp']}
                        - **Utilizator:** {entry['utilizator']}
                        """)
                        st.write("📊 Valori prezise:")
                        st.line_chart(entry["predictii"])

                st.markdown("---")
                confirm = st.checkbox("✅ Confirm ștergerea definitivă")
                if st.button("🗑️ Șterge istoricul meu de predicții") and confirm:
                    try:
                        filtrate = [p for p in toate_pred if p.get("utilizator") != st.session_state["username"]]

                        with open("istoric.yaml", "w") as f:
                            yaml.dump({"predictii": filtrate}, f)

                        st.success("✅ Istoricul tău a fost șters.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"⚠️ Eroare la ștergere: {e}")
        except Exception as e:
            st.error(f"⚠️ Eroare la citirea fișierului: {e}")


elif page == "Joc de predicții":
    from datetime import date, timedelta
    import pandas as pd
    import requests
    import yaml
    import os

    st.header("🎮 Joc de predicții - Ghiceste cursul de mâine!")
    st.header("Valutele ce participa la acest joc sunt: USD, EUR, AUD, BGN, BRL, CHF, CNY, INR")

    try:
        df = pd.read_csv("currency_data.csv", index_col=0)
        df.index = pd.to_datetime(df.index)

        valute_permise = ["USD", "EUR", "AUD", "BGN", "BRL", "CHF", "CNY", "INR"]
        valute = [
            col for col in df.columns
            if any(f"{base}/{quote}" == col for base in ["USD", "EUR"] for quote in valute_permise)
        ]

        if not valute:
            st.warning("⚠️ Nu există valute disponibile din lista permisă.")
            st.stop()

    except Exception as e:
        st.warning(f"Nu s-au putut încărca datele valutare: {e}")
        st.stop()

    st.subheader("🔢 Trimite predicția ta pentru mâine")
    valuta = st.selectbox("Alege valuta", valute)
    predictie = st.number_input("Introdu cursul estimat pentru mâine", format="%.4f", step=0.0001)
    model_ai = st.selectbox("Model AI de comparat", ["ARIMA", "RNN"])

    if st.button("🚀 Trimite predicția"):
        success = adauga_predictie_user(
            username=st.session_state["username"],
            valuta=valuta,
            predictie_user=predictie,
            model_ai=model_ai
        )
        if success:
            st.success("✅ Predicția ta a fost salvată și AI-ul a fost provocat!")

            # 🔁 Verificăm și completăm scorul pentru ziua de ieri
            ieri = str(date.today() - timedelta(days=1))
            try:
                url = f"https://api.frankfurter.app/{ieri}?from=USD"
                resp = requests.get(url)
                if resp.status_code == 200:
                    rates = resp.json()["rates"]

                    if os.path.exists("joc_predictii.yaml"):
                        with open("joc_predictii.yaml", "r") as f:
                            data = yaml.safe_load(f) or []

                        valute_ieri = set(
                            entry["valuta"] for entry in data
                            if entry["data"] == ieri and entry["username"] == st.session_state["username"]
                            and entry["valoare_real"] is None
                        )

                        for valuta in valute_ieri:
                            cod_final = valuta.split("/")[-1]
                            if cod_final in rates:
                                valoare_real = rates[cod_final]
                                actualizeaza_valoare_reală(ieri, valuta, valoare_real)

                        if calculeaza_scoruri():
                            st.success("🧠 Scorurile au fost actualizate pentru ziua de ieri!")
            except Exception as e:
                st.warning(f"⚠️ Eroare la verificarea scorurilor: {e}")
        else:
            st.warning("⚠️ Ai făcut deja o predicție pentru această valută azi.")

    st.markdown("---")
    st.subheader("🏆 Clasament utilizatori")

    clasament = genereaza_clasament()
    if clasament:
        for i, (user, scor) in enumerate(clasament, 1):
            st.markdown(f"**{i}. {user}** — {scor} puncte")
    else:
        st.info("Nu există încă scoruri înregistrate.")

    st.markdown("---")
    st.subheader("🎖️ Insignele tale")

    insigne = insigne_utilizator(st.session_state["username"])
    if insigne:
        st.write(" ".join(insigne))
    else:
        st.info("Nu ai încă insigne. Fă predicții zilnice și bate AI-ul pentru a le câștiga!")

    if os.path.exists("joc_predictii.yaml"):
        with open("joc_predictii.yaml", "r") as f:
            data = yaml.safe_load(f) or []

        ieri = str(date.today() - timedelta(days=1))
        pred = [
            e for e in data
            if e["username"] == st.session_state["username"] and e["data"] == ieri
            and e["valoare_real"] is not None
        ]

        if pred:
            st.subheader("📊 Rezultatul tău de ieri:")
            for e in pred:
                st.markdown(f"""
                🔁 Valută: **{e['valuta']}**
                - 🎯 Valoare reală: `{e['valoare_real']}`
                - 👤 Tu: `{e['predictie_user']}` → {e['scor_user']} puncte  
                - 🤖 AI: `{e['predictie_ai']}` → {e['scor_ai']} puncte
                """)
                if e["scor_user"] > e["scor_ai"]:
                    st.success("🏆 Ai  învins AI-ul!")
                elif e["scor_user"] < e["scor_ai"]:
                    st.warning("🤖 AI-ul a fost mai bun ieri.")
                else:
                    st.info("🤝 Egalitate!")


elif page == "Delogare":
    st.info("Te-ai delogat.")
    if st.button("Confirmă delogarea"):
        st.session_state.clear()
        st.rerun()
