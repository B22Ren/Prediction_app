import warnings
import pandas as pd
import numpy as np
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRAINING_PERCENTAGE = 0.8
NUMBER_OF_PREVIOUS_DATA_POINTS = 3
LENGTH_DATA_SET = 0
np.random.seed(7)
TRAINING_SET_LENGTH = 0
TESTING_SET_LENGTH = 0

def training_testing_buckets(raw_data, training_percentage):
    global TRAINING_SET_LENGTH, TESTING_SET_LENGTH
    TRAINING_SET_LENGTH = int(LENGTH_DATA_SET * training_percentage)
    TESTING_SET_LENGTH = LENGTH_DATA_SET - TRAINING_SET_LENGTH
    training_set = raw_data[0:TRAINING_SET_LENGTH]
    testing_set = raw_data[TRAINING_SET_LENGTH:LENGTH_DATA_SET]
    return training_set, testing_set

def modify_data_set_rnn(training_set, testing_set):
    def create_sequences(data):
        actual, predict = [], []
        for i in range(len(data) - NUMBER_OF_PREVIOUS_DATA_POINTS - 1):
            actual.append(data[i: i + NUMBER_OF_PREVIOUS_DATA_POINTS])
            predict.append(data[i + NUMBER_OF_PREVIOUS_DATA_POINTS])
        return np.array(actual), np.array(predict)

    return (*create_sequences(training_set), *create_sequences(testing_set))

def load_data_set(currency):
    try:
        df = pd.read_csv("currency_data.csv", header=0, index_col=0, parse_dates=True)

        if currency not in df.columns:
            raise ValueError(f"Valuta '{currency}' nu a fost găsită în fișier.")

        raw_data = df[currency].dropna().tolist()
        global LENGTH_DATA_SET
        LENGTH_DATA_SET = len(raw_data)

        dates = df.index.tolist()
        return raw_data, dates
    except Exception as e:
        raise RuntimeError(f"Eroare la încărcarea datasetului: {e}")


def build_rnn_model(train_actual, train_predict):
    train_actual = train_actual.reshape((train_actual.shape[0], train_actual.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, input_shape=(NUMBER_OF_PREVIOUS_DATA_POINTS, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_actual, train_predict, epochs=100, batch_size=16, verbose=0)
    return model

def predict_rnn(model, train_actual, test_actual):
    train_actual = train_actual.reshape((train_actual.shape[0], train_actual.shape[1], 1))
    test_actual = test_actual.reshape((test_actual.shape[0], test_actual.shape[1], 1))
    return model.predict(train_actual), model.predict(test_actual)

def plot_rnn(currency, raw_data, training_predict, testing_predict, scaler, file_name, forecast_future=None):
    training_real = scaler.inverse_transform(training_predict)
    testing_real = scaler.inverse_transform(testing_predict)

    plt.figure(figsize=(10, 6))
    plt.plot(raw_data, label="Valori reale", color="blue")
    plt.plot(range(NUMBER_OF_PREVIOUS_DATA_POINTS, NUMBER_OF_PREVIOUS_DATA_POINTS + len(training_real)),
             training_real[:, 0], label="Predicții antrenare", color="green")
    plt.plot(range(TRAINING_SET_LENGTH + NUMBER_OF_PREVIOUS_DATA_POINTS,
                   TRAINING_SET_LENGTH + NUMBER_OF_PREVIOUS_DATA_POINTS + len(testing_real)),
             testing_real[:, 0], label="Predicții testare", color="red")

    if forecast_future:
        future_x = list(range(TRAINING_SET_LENGTH + NUMBER_OF_PREVIOUS_DATA_POINTS + len(testing_real),
                              TRAINING_SET_LENGTH + NUMBER_OF_PREVIOUS_DATA_POINTS + len(testing_real) + len(forecast_future)))
        plt.plot(future_x, forecast_future, label="Predicții viitoare", color="orange", linestyle="--")

    plt.title(f"Predicții RNN - USD/{currency.upper()}")
    plt.xlabel("Număr de zile")
    plt.ylabel(f"Valoare USD/{currency.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def predict_future_days(model, last_window, days, scaler):
    future_preds = []
    window = last_window.copy()
    for _ in range(days):
        input_seq = np.array(window).reshape((1, NUMBER_OF_PREVIOUS_DATA_POINTS, 1))
        next_pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
        next_pred = scaler.inverse_transform([[next_pred_scaled]])[0, 0]
        future_preds.append(next_pred)
        window = np.append(window[1:], next_pred_scaled)
    return future_preds
def evaluate_rnn_performance(y_true_scaled, y_pred_scaled, scaler):
    
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "mae": mae, "r2": r2}

def rnn_model(currency, forecast_days=10):
    raw_data, dates = load_data_set(currency)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1)).flatten()
    training_set, testing_set = training_testing_buckets(scaled_data, TRAINING_PERCENTAGE)
    train_actual, train_predict, test_actual, test_predict = modify_data_set_rnn(training_set, testing_set)

    model = build_rnn_model(train_actual, train_predict)
    training_predict, testing_predict = predict_rnn(model, train_actual, test_actual)

   
    train_metrics = evaluate_rnn_performance(train_predict, training_predict, scaler)
    test_metrics = evaluate_rnn_performance(test_predict, testing_predict, scaler)

   
    print("=== Performanță RNN - Set Antrenare ===")
    print(f"MAE: {train_metrics['mae']:.4f}")
    print(f"MSE: {train_metrics['mse']:.4f}")
    print(f"R²:  {train_metrics['r2']:.4f}\n")

    print("=== Performanță RNN - Set Testare ===")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"MSE: {test_metrics['mse']:.4f}")
    print(f"R²:  {test_metrics['r2']:.4f}")

    
    last_window = scaled_data[-NUMBER_OF_PREVIOUS_DATA_POINTS:]
    future_predictions = predict_future_days(model, last_window, forecast_days, scaler)
    plot_rnn(currency, raw_data, training_predict, testing_predict, scaler, "predictie_rnn.pdf", future_predictions)

    return {
        "raw_data": raw_data,
        "training_predict": training_predict,
        "testing_predict": testing_predict,
        "future_predictions": future_predictions,
        "scaler": scaler,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }


warnings.filterwarnings("ignore")
