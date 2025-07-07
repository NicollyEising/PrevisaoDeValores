from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# Importações do seu código existente
from main import (
    prepare_data, train_lstm, predict_lstm, train_xgb,
    mape, LSTMModel, RandomForestRegressor
)

app = FastAPI(title="Bitcoin Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou especifique ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Treinamento único na inicialização
X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y, df = prepare_data()

# Treinar modelos
lstm_model, device = train_lstm(X_train, y_train, X_val, y_val)
lstm_model.eval()
xgb_model, _ = train_xgb(np.vstack([X_train, X_val]), np.vstack([y_train, y_val]), X_test)

# Ensemble

lstm_preds = predict_lstm(lstm_model, np.vstack([X_val, X_test]), scaler_y)
xgb_preds = scaler_y.inverse_transform(xgb_model.predict(np.vstack([X_val, X_test])))
y_true_comb = scaler_y.inverse_transform(np.vstack([y_val, y_test]))

n = min(len(lstm_preds), len(xgb_preds), len(y_true_comb))
lstm_preds, xgb_preds, y_true = lstm_preds[-n:], xgb_preds[-n:], y_true_comb[-n:]
meta_X = np.hstack([lstm_preds, xgb_preds])
mask = ~np.isnan(meta_X).any(axis=1) & ~np.isnan(y_true).any(axis=1)

meta_X_clean, y_clean = meta_X[mask], y_true[mask]
meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
meta_model.fit(meta_X_clean, y_clean)

# ----------------------------
# MODELOS DE DADOS PARA A API
# ----------------------------
class PredictionResponse(BaseModel):
    close: float
    high: float
    low: float
    mape_close: float
    mape_high: float
    mape_low: float




@app.get("/predict", response_model=PredictionResponse)
def predict_next_day():
    # --- código para obter next_pred como antes ---
    time_steps = 30
    full_X = np.vstack([X_val, X_test])
    last_window = full_X[-time_steps:]

    with torch.no_grad():
        lstm_input = torch.tensor(last_window.reshape(1, time_steps, -1), dtype=torch.float32).to(device)
        next_lstm_scaled = lstm_model(lstm_input).cpu().numpy()
    next_lstm = scaler_y.inverse_transform(next_lstm_scaled)

    last_feat = last_window[-1].reshape(1, -1)
    next_xgb_scaled = xgb_model.predict(last_feat)
    next_xgb = scaler_y.inverse_transform(next_xgb_scaled)

    next_meta_input = np.hstack([next_lstm, next_xgb])
    next_pred = meta_model.predict(next_meta_input).flatten()

    # Calcular MAPE usando os dados históricos (exemplo: últimos y_clean e preds)
    # Aqui usando os dados calculados na inicialização
    errs = mape(y_clean, meta_model.predict(meta_X_clean))

    return PredictionResponse(
        close=float(next_pred[0]),
        high=float(next_pred[1]),
        low=float(next_pred[2]),
        mape_close=float(errs[0]),
        mape_high=float(errs[1]),
        mape_low=float(errs[2]),
    )

class ChartDataResponse(BaseModel):
    dates: List[str]
    real_close: List[float]
    lstm_close: List[float]
    xgb_close: List[float]
    ensemble_close: List[float]

@app.get("/chart-data", response_model=ChartDataResponse)
def get_chart_data():
    # Repetir processamento para alinhar e limpar dados
    lstm_preds = predict_lstm(lstm_model, np.vstack([X_val, X_test]), scaler_y, device=device)
    xgb_preds_raw = xgb_model.predict(np.vstack([X_val, X_test]))
    xgb_preds = scaler_y.inverse_transform(xgb_preds_raw)
    y_true_comb = scaler_y.inverse_transform(np.vstack([y_val, y_test]))

    n = min(len(lstm_preds), len(xgb_preds), len(y_true_comb))
    lstm_preds, xgb_preds, y_true = lstm_preds[-n:], xgb_preds[-n:], y_true_comb[-n:]

    meta_X = np.hstack([lstm_preds, xgb_preds])
    mask = ~np.isnan(meta_X).any(axis=1) & ~np.isnan(y_true).any(axis=1)

    meta_X_clean, y_clean = meta_X[mask], y_true[mask]

    ensemble_preds = meta_model.predict(meta_X_clean)

    # Datas correspondentes aos dados limpos
    dates = df.index[-len(y_true):][-n:][mask]

    # Converter datas para string para JSON
    dates_str = dates.strftime("%Y-%m-%d").tolist()

    return ChartDataResponse(
        dates=dates_str,
        real_close=y_clean[:, 0].tolist(),
        lstm_close=lstm_preds[mask, 0].tolist(),
        xgb_close=xgb_preds[mask, 0].tolist(),
        ensemble_close=ensemble_preds[:, 0].tolist()
    )