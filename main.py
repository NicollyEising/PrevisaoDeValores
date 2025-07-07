import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series, period=20):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + 2*std, sma - 2*std

def stochastic_oscillator(df, k_window=14, d_window=3):
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    percent_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    percent_d = percent_k.rolling(window=d_window).mean()
    return percent_k, percent_d

def average_true_range(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def add_technical_indicators(df):
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Upper_BB'], df['Lower_BB'] = bollinger_bands(df['Close'])
    df['Stoch_%K'], df['Stoch_%D'] = stochastic_oscillator(df)
    df['ATR'] = average_true_range(df)
    df['ROC'] = df['Close'].pct_change(10) * 100

    for lag in range(1, 4):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'High_lag_{lag}']  = df['High'].shift(lag)
        df[f'Low_lag_{lag}']   = df['Low'].shift(lag)

    df['DayOfWeek'] = df.index.dayofweek
    df['Month']     = df.index.month
    df.dropna(inplace=True)
    return df


class LSTMDataset(Dataset):
    def __init__(self, features, targets, time_steps=30):
        self.X, self.y = self._create_sequences(features, targets, time_steps)

    @staticmethod
    def _create_sequences(features, targets, time_steps):
        Xs, ys = [], []
        for i in range(len(features) - time_steps):
            Xs.append(features[i:(i + time_steps)])
            ys.append(targets[i + time_steps])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = self.norm(h)
        h = self.dropout(h)
        return self.fc(h)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) /
               np.where(y_true == 0, np.nan, y_true)), axis=0) * 100

def prepare_data():
    df = yf.download("BTC-USD", start="2015-01-01", end="2025-06-25", auto_adjust=True)
    df = add_technical_indicators(df)
    targets = df[['Close','High','Low']].shift(-1)
    df.dropna(inplace=True)

    X = df.drop(['Close','High','Low'], axis=1).values
    y = targets.loc[df.index].values

    scaler_X = MinMaxScaler()
    scaler_y = RobustScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, shuffle=False)

    logging.info(f"Shapes — X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y, df

def train_lstm(X_train, y_train, X_val, y_val, time_steps=30,
               epochs=100, batch_size=32, device=None):
    ds_train = LSTMDataset(X_train, y_train, time_steps)
    ds_val   = LSTMDataset(X_val,   y_val,   time_steps)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size=X_train.shape[1])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val, patience, wait = np.inf, 10, 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in loader_val:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())

        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        scheduler.step()
        logging.info(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")

        if val_loss < best_val:
            best_val, best_state, wait = val_loss, model.state_dict(), 0
        else:
            wait += 1
            if wait >= patience:
                logging.info("Early stopping ativado.")
                break

    model.load_state_dict(best_state)
    return model, device

def predict_lstm(model, X, scaler_y, time_steps=30, device=None):
    ds = LSTMDataset(X, np.zeros((len(X), 3)), time_steps)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    return scaler_y.inverse_transform(np.vstack(preds))

def train_xgb(X_trainval, y_trainval, X_test):
    xgb = MultiOutputRegressor(XGBRegressor(
        n_estimators=300, learning_rate=0.02, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1,
        random_state=42, verbosity=0
    ))
    xgb.fit(X_trainval, y_trainval)
    return xgb, xgb.predict(X_test)

def main():
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y, df = prepare_data()

    lstm_model, device = train_lstm(X_train, y_train, X_val, y_val)
    lstm_preds = predict_lstm(lstm_model, np.vstack([X_val, X_test]), scaler_y)

    y_true_comb = scaler_y.inverse_transform(np.vstack([y_val, y_test]))

    xgb_model, xgb_preds = train_xgb(
        np.vstack([X_train, X_val]), np.vstack([y_train, y_val]),
        np.vstack([X_val, X_test])
    )
    xgb_preds = scaler_y.inverse_transform(xgb_preds)

    # alinhar comprimentos
    n = min(len(lstm_preds), len(xgb_preds), len(y_true_comb))
    lstm_preds, xgb_preds, y_true = (
        lstm_preds[-n:], xgb_preds[-n:], y_true_comb[-n:]
    )

    # filtrar linhas com NaN em meta_X ou y_true
    meta_X = np.hstack([lstm_preds, xgb_preds])
    mask = ~np.isnan(meta_X).any(axis=1) & ~np.isnan(y_true).any(axis=1)
    meta_X_clean, y_clean = meta_X[mask], y_true[mask]

    # ensemble com RandomForest
    meta = RandomForestRegressor(n_estimators=100, random_state=42)
    meta.fit(meta_X_clean, y_clean)

    ensemble = meta.predict(meta_X_clean)


    for name, pred in zip(['LSTM','XGBoost','Ensemble'], 
                          [lstm_preds[mask], xgb_preds[mask], ensemble]):
        errs = mape(y_clean, pred)
        logging.info(f"{name:8s} | Close: {errs[0]:5.2f}% | "
                     f"High: {errs[1]:5.2f}% | Low: {errs[2]:5.2f}%")
        

    time_steps = 30  

    full_X = np.vstack([X_val, X_test])
    last_window = full_X[-time_steps:]  

    # Previsão LSTM
    with torch.no_grad():
        lstm_input = torch.tensor(last_window.reshape(1, time_steps, -1),
                                  dtype=torch.float32).to(device)
        next_lstm_scaled = lstm_model(lstm_input).cpu().numpy()
    next_lstm = scaler_y.inverse_transform(next_lstm_scaled)

    # Previsão XGBoost
    last_feat = last_window[-1].reshape(1, -1)
    next_xgb_scaled = xgb_model.predict(last_feat)
    next_xgb = scaler_y.inverse_transform(next_xgb_scaled)

    # Combinação Ensemble
    next_meta_input = np.hstack([next_lstm, next_xgb])
    next_pred = meta.predict(next_meta_input).flatten()

    # Exibição
    print(f"Previsão para o próximo dia:")
    print(f"  Close: {next_pred[0]:.2f}")
    print(f"  High : {next_pred[1]:.2f}")
    print(f"  Low  : {next_pred[2]:.2f}")


    # plotagem
    dates = df.index[-len(y_clean):]
    plt.figure(figsize=(12,6))
    plt.plot(dates, y_clean[:,0], label='Real - Close', linewidth=2)
    plt.plot(dates, lstm_preds[mask,0], '--', label='LSTM - Close')
    plt.plot(dates, xgb_preds[mask,0], '--', label='XGB - Close')
    plt.plot(dates, ensemble[:,0], '--', label='Ensemble - Close')
    plt.title("Previsões vs Real – Preço de Fechamento")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


