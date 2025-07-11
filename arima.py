import pandas as pd
import warnings
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
import os

TRAINING_PERCENTAGE = 0.7
TESTING_PERCENTAGE = 1 - TRAINING_PERCENTAGE
LENGTH_DATA_SET = 0
TRAINING_SET_LENGTH = 0
TESTING_SET_LENGTH = 0


def training_testing_buckets(raw_data, training_percentage, testing_percentage):
    global TRAINING_SET_LENGTH, TESTING_SET_LENGTH
    TRAINING_SET_LENGTH = int(LENGTH_DATA_SET * training_percentage)
    TESTING_SET_LENGTH = LENGTH_DATA_SET - TRAINING_SET_LENGTH
    training_set = raw_data[0:TRAINING_SET_LENGTH]
    testing_set = raw_data[TRAINING_SET_LENGTH:LENGTH_DATA_SET]
    return training_set, testing_set


def plot_arima(currency, testing_actual, testing_predict, file_name, training_series=None):
    plt.figure(figsize=(10, 6))

    total_length = len(training_series) + len(testing_actual) if training_series else 0
    x_train = list(range(len(training_series)))
    x_test_actual = list(range(len(training_series), total_length))
    x_test_predict = x_test_actual

    if training_series:
        plt.plot(x_train, training_series, label="Date antrenare", color="dodgerblue", linewidth=2)
    plt.plot(x_test_actual, testing_actual, label="Date reale (test)", color="black", linestyle='--', linewidth=2)
    plt.plot(x_test_predict, testing_predict, label="Predicții ARIMA", color="limegreen", linewidth=2)

    plt.xlabel("Număr de zile", fontsize=12)
    plt.ylabel(f"Valoare USD/{currency.upper()}", fontsize=12)
    plt.title(f"USD/{currency.upper()} - Antrenare, Testare și Predicții ARIMA", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def load_data_set(currency):
    try:
        data_set_frame = pd.read_csv("currency_data.csv", header=0, index_col=0)
        data_set_frame.index = pd.to_datetime(data_set_frame.index)

        if currency not in data_set_frame.columns:
            raise ValueError(f"Valuta '{currency}' nu a fost găsită în fișier.")

        raw_data = data_set_frame[currency].dropna().tolist()

        global LENGTH_DATA_SET
        LENGTH_DATA_SET = len(raw_data)

        dates = data_set_frame.index.tolist()
        return raw_data, dates
    except Exception as e:
        raise RuntimeError(f"Eroare la încărcarea datasetului: {e}")


def evaluate_performance_arima(testing_actual, testing_predict):
    mse = mean_squared_error(testing_actual, testing_predict)
    mae = mean_absolute_error(testing_actual, testing_predict)
    r2 = r2_score(testing_actual, testing_predict)
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2
    }


def build_model_predict_arima(training_set, testing_set):
    global TESTING_SET_LENGTH
    TESTING_SET_LENGTH = len(testing_set)
    testing_predict = []
    training_predict = list(training_set)

    arima_model = None  

    if TESTING_SET_LENGTH == 0:
      
        arima = ARIMA(training_predict, order=(5, 1, 0))
        arima_model = arima.fit()
        return arima_model, [], training_predict

    for testing_set_index in range(TESTING_SET_LENGTH):
        arima = ARIMA(training_predict, order=(5, 1, 0))
        arima_model = arima.fit()
        forecasting = arima_model.forecast(steps=1)[0]
        testing_predict.append(forecasting)
        training_predict.append(testing_set[testing_set_index])

    return arima_model, testing_predict, training_predict


def forecast_future_days(arima_model_fit, training_predict_series, days_to_predict):
    future_predictions = []
    training_series = list(training_predict_series)

    for _ in range(days_to_predict):
        arima = ARIMA(training_series, order=(5, 1, 0))
        arima_model_fit = arima.fit()
        next_forecast = arima_model_fit.forecast(steps=1)[0]
        future_predictions.append(next_forecast)
        training_series.append(next_forecast)

    return future_predictions


def arima_model(currency):
    raw_data, dates = load_data_set(currency)
    training_actual, testing_actual = training_testing_buckets(raw_data, TRAINING_PERCENTAGE, TESTING_PERCENTAGE)
    arima_model_fit, testing_predict, training_predict_series = build_model_predict_arima(training_actual, testing_actual)
    metrics = evaluate_performance_arima(testing_actual, testing_predict)
    plot_arima(currency, testing_actual, testing_predict, "predictie_arima.pdf", training_series=training_actual)

    print("=== Performanță ARIMA - Set Testare ===")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"R²:  {metrics['r2']:.4f}")

    return {
        "raw_data": raw_data,
        "dates": dates,
        "training": training_actual,
        "testing": testing_actual,
        "model": arima_model_fit,
        "testing_predict": testing_predict,
        "training_predict_series": training_predict_series,
        **metrics
    }


warnings.filterwarnings("ignore")
