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


#Indicadores técnicos
#calcula rsi, onde separa os ganhos e perdas diarios, e o rs é o resultado entre 0 e 100 que acima de 70 é sobrecomprado e abaixo de 30 é sobrevendido
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(series, period=20):
    #Calcula media movel
    sma = series.rolling(window=period).mean()
    #Calcula desvio padrao
    std = series.rolling(window=period).std()
    #Calcula bandas superior e inferior utilizando a media movel e o desvio padrao
    return sma + 2*std, sma - 2*std

def stochastic_oscillator(df, k_window=14, d_window=3):
    #Calcula o menor valor de baixa
    low_min = df['Low'].rolling(window=k_window).min()
    #Calcula o maior valor de alta
    high_max = df['High'].rolling(window=k_window).max()
    #Calcula o %K, que é a porcentagem do fechamento atual em relação ao intervalo de alta e baixa
    percent_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    #Calcula o %D, que é a media movel do %K
    percent_d = percent_k.rolling(window=d_window).mean()
    #Retorna %K e %D que acima de 80 é sobrecomprado e abaixo de 20 é sobrevendido
    return percent_k, percent_d

#ATR mede a volatilidade do ativo
def average_true_range(df, period=14):
    #Calcula o intervalo da barra (high - low) para o período atual
    high_low = df['High'] - df['Low']
    #Calcula o intervalo entre o fechamento do período anterior e o máximo e mínimo do período atual
    high_close = (df['High'] - df['Close'].shift()).abs()
    #Calcula a diferença absoluta entre a mínima do dia atual e o fechamento do dia anterior
    low_close = (df['Low'] - df['Close'].shift()).abs()
    #Calcula o True Range (TR) como o máximo entre os três intervalos calculados
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    #Calcula a média móvel do True Range (TR) para o período especificado
    return tr.rolling(window=period).mean()

# Adiciona indicadores técnicos ao DataFrame
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

#Dataset e modelo LSTM para previsão de séries temporais
#Cria uma subclasse de Dataset para alimentar modelos durante o treinamento
class LSTMDataset(Dataset):
    # Inicializa o dataset com as sequências de características (preços, indicadores tecnicos, etc.) e alvos
    def __init__(self, features, targets, time_steps=30):
        #Constroi os dados em sequencias, e armazena em self.X (entradas) e self.y (alvos)
        self.X, self.y = self._create_sequences(features, targets, time_steps)

    @staticmethod
    def _create_sequences(features, targets, time_steps):
        Xs, ys = [], []
        #Pega sempre um conjunto de 30 números seguidos
        #E pega o número que vem logo depois como a resposta esperada
        for i in range(len(features) - time_steps):
            Xs.append(features[i:(i + time_steps)])
            ys.append(targets[i + time_steps])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    # Retorna o tamanho do dataset
    def __len__(self):
        return len(self.X)
    #Permite indexar o dataset como uma lista
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
# nn.Module base class para construir modelos de aprendizado profundo
class LSTMModel(nn.Module):
    # hidden_size=128 número de neurônios nas camadas ocultas da LSTM
    # num_layers=3 número de camadas LSTM empilhadas
    # output_size=3 número de variáveis preditas (ex: previsão de 3 valores → output_size=3)
    # dropout=0.2 taxa de dropout para regularização
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=3, dropout=0.2):
        #Inicializa a classe base (nn.Module)
        super().__init__()
        #Cria uma camada LSTM com os parâmetros especificados aneteriormente
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        # Aplica normalização camada a camada nos vetores ocultos da LSTM
        self.norm = nn.LayerNorm(hidden_size)
        # plica dropout (regularização) antes da camada final para evitar overfitting
        self.dropout = nn.Dropout(dropout)
        # Cria uma camada totalmente conectada para mapear os vetores ocultos para as saídas
        self.fc = nn.Linear(hidden_size, output_size)

    # define o fluxo dos dados pela rede durante o treinamento.
    def forward(self, x):
        # Passa o dado x pela camada LSTM. A saída out contém a resposta da LSTM
        out, _ = self.lstm(x)
        # Pega apenas a última saída da sequência
        h = out[:, -1, :]
        # Aplica normalização, dropout e a camada totalmente conectada
        h = self.norm(h)
        # Normaliza a saída da LSTM
        h = self.dropout(h)
        # Passa pela camada totalmente conectada para obter a previsão final
        return self.fc(h)

#Mape avalia o erro percentual médio entre valores verdadeiros e previstos
def mape(y_true, y_pred):
    # calcula o erro absoluto entre valor real e previsto
    return np.mean(np.abs((y_true - y_pred) /
            # evita divisão por zero substituindo valores reais iguais a zero por NaN
               np.where(y_true == 0, np.nan, y_true)), axis=0) * 100

# Pré-processamento dos dados
def prepare_data():
    #Faz o download dos dados históricos diários 
    df = yf.download("BTC-USD", start="2015-01-01", end="2025-06-25", auto_adjust=True)
    # adiciona indicadores técnicos ao DataFrame
    df = add_technical_indicators(df)
    # Cria o conjunto de alvos para previsão, com os preços de fechamento, máxima e mínima do próximo dia
    # shift(-1) desloca os valores para cima, de modo que o valor do dia seguinte seja o alvo
    targets = df[['Close','High','Low']].shift(-1)
    # Remove as linhas com valores ausentes (NaN)
    df.dropna(inplace=True)

    # Separa as colunas de características (X) e alvos (y)
    X = df.drop(['Close','High','Low'], axis=1).values
    y = targets.loc[df.index].values

    # Normaliza as características e alvos usando MinMaxScaler e RobustScaler
    # MinMaxScaler escala os dados entre 0 e 1, útil para redes neurais
    scaler_X = MinMaxScaler()
    # RobustScaler é menos sensível a outliers, útil para dados financeiros
    scaler_y = RobustScaler()

    # Aplica a normalização
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Separa os dados em conjuntos treino+validação (80%) e teste (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, shuffle=False)
    # Divide o conjunto treino+validação em treino (90%) e validação (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, shuffle=False)

    logging.info(f"Shapes — X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    # Retorna os dados divididos
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y, df

# Treinamento de um modelo LSTM
# Parâmetros de configuração: tamanho da janela time_steps, número de épocas, tamanho do batch e dispositivo para treinamento.
def train_lstm(X_train, y_train, X_val, y_val, time_steps=30,
               epochs=100, batch_size=32, device=None):
    # Cria dois data loaders, que são estruturas responsáveis por entregar os dados em partes menores
    ds_train = LSTMDataset(X_train, y_train, time_steps)
    ds_val   = LSTMDataset(X_val,   y_val,   time_steps)
    
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)
    #Inicializa o modelo LSTM com o número correto de features (input size).
    model = LSTMModel(input_size=X_train.shape[1])
    # Se nenhum dispositivo for especificado, usa CUDA se disponível, caso contrário CPU
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # define a função de perda (ou erro) usada para treinar o modelo e otimizador AdamW para atualização dos pesos, com taxa de aprendizado 0.001
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # Scheduler para reduzir a taxa de aprendizado a cada 20 épocas, manter essa taxa alta pode causar oscilações ou impedir que o modelo refine bem os pesos próximos do mínimo do erro
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # controle de early stopping, patience : número de épocas sem melhora para parar.
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


        # Avalia o desempenho do modelo no conjunto de validação
        model.eval()  # Coloca o modelo em modo de avaliação (desativa dropout, etc.)
        val_losses = []  # Lista para armazenar as perdas de validação

        with torch.no_grad():  # Desativa o cálculo de gradientes (mais rápido e economiza memória)
            for xb, yb in loader_val:  # Percorre os dados de validação em batches
                xb, yb = xb.to(device), yb.to(device)  # Move os dados para GPU ou CPU
                val_losses.append(criterion(model(xb), yb).item())  # Calcula e armazena a perda do batch

        # Calcula a média das perdas de treinamento e validação
        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        scheduler.step()
        logging.info(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")

        #Implementa early stopping
        if val_loss < best_val:
            best_val, best_state, wait = val_loss, model.state_dict(), 0
        else:
            wait += 1
            if wait >= patience:
                logging.info("Early stopping ativado.")
                break
    # Carrega os pesos do melhor modelo encontrado durante o treinamento
    model.load_state_dict(best_state)
    return model, device

# Realiza previsões com um modelo LSTM já treinado
def predict_lstm(model, X, scaler_y, time_steps=30, device=None):
    # Cria um dataset do tipo LSTMDataset com os dados de entrada X
    ds = LSTMDataset(X, np.zeros((len(X), 3)), time_steps)
    # Prepara o carregamento dos dados em batches
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    model.eval()
    preds = []
    # Previsão em modo de avaliação, sem calcular gradientes
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    # Retorno das previsões, aplicando a inversão da normalização dos alvos
    return scaler_y.inverse_transform(np.vstack(preds))

# Treinamento de um modelo XGBoost para regressão multivariada
def train_xgb(X_trainval, y_trainval, X_test):
    # Cria um modelo XGBoost com parâmetros específicos para regressão
    xgb = MultiOutputRegressor(XGBRegressor(
        n_estimators=300, learning_rate=0.02, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1,
        random_state=42, verbosity=0
    ))
    # Treina o modelo com os dados de treino+validação
    xgb.fit(X_trainval, y_trainval)
    # Retorna o modelo treinado e as predições geradas para X_test.
    return xgb, xgb.predict(X_test)

def main():
    #Chama a função que: Baixa os dados históricos, Gera os indicadores técnicos, Cria os conjuntos de treino, validação e teste, Normaliza os dados, Retorna os scalers e o DataFrame original.
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y, df = prepare_data()

    # Treina o modelo LSTM e obtém as previsões
    lstm_model, device = train_lstm(X_train, y_train, X_val, y_val)
    lstm_preds = predict_lstm(lstm_model, np.vstack([X_val, X_test]), scaler_y)
    # Concatena os alvos de validação e teste e os converte de volta à escala real.
    y_true_comb = scaler_y.inverse_transform(np.vstack([y_val, y_test]))

    # Treina o modelo XGBoost e obtém as previsões
    xgb_model, xgb_preds = train_xgb(
        np.vstack([X_train, X_val]), np.vstack([y_train, y_val]),
        np.vstack([X_val, X_test])
    )
    xgb_preds = scaler_y.inverse_transform(xgb_preds)

    # Alinhamento e limpeza dos dados
    # Garante que as três matrizes tenham o mesmo comprimento final.
    n = min(len(lstm_preds), len(xgb_preds), len(y_true_comb))
    lstm_preds, xgb_preds, y_true = (
        lstm_preds[-n:], xgb_preds[-n:], y_true_comb[-n:]
    )

    # filtrar linhas com NaN tanto nas predições quanto nos alvos.
    meta_X = np.hstack([lstm_preds, xgb_preds])
    mask = ~np.isnan(meta_X).any(axis=1) & ~np.isnan(y_true).any(axis=1)
    meta_X_clean, y_clean = meta_X[mask], y_true[mask]

    # Treinamento do modelo ensemble (Random Forest)
    # Treina um modelo Random Forest para combinar as saídas do LSTM e do XGBoost
    meta = RandomForestRegressor(n_estimators=100, random_state=42)
    meta.fit(meta_X_clean, y_clean)
    ensemble = meta.predict(meta_X_clean)

    #Calcula e imprime o erro percentual médio absoluto (MAPE) para cada modelo
    for name, pred in zip(['LSTM','XGBoost','Ensemble'], 
                          [lstm_preds[mask], xgb_preds[mask], ensemble]):
        errs = mape(y_clean, pred)
        logging.info(f"{name:8s} | Close: {errs[0]:5.2f}% | "
                     f"High: {errs[1]:5.2f}% | Low: {errs[2]:5.2f}%")

    time_steps = 30
    # Prepara os dados para previsão do próximo dia
    full_X = np.vstack([X_val, X_test])
    last_window = full_X[-time_steps:]  # Prepara a última janela de time_steps dias

    # Gera a previsão do LSTM para o próximo dia
    with torch.no_grad():
        lstm_input = torch.tensor(last_window.reshape(1, time_steps, -1),
                                  dtype=torch.float32).to(device)
        next_lstm_scaled = lstm_model(lstm_input).cpu().numpy()
    next_lstm = scaler_y.inverse_transform(next_lstm_scaled)

    # Gera a previsão do XGBoost para o próximo dia.
    last_feat = last_window[-1].reshape(1, -1)
    next_xgb_scaled = xgb_model.predict(last_feat)
    next_xgb = scaler_y.inverse_transform(next_xgb_scaled)

    # O modelo ensemble faz a previsão final com base nas saídas dos dois modelos anteriores.
    next_meta_input = np.hstack([next_lstm, next_xgb])
    next_pred = meta.predict(next_meta_input).flatten()

    # Imprime a previsão para fechamento, máxima e mínima do próximo dia.
    print(f"Previsão para o próximo dia:")
    print(f"  Close: {next_pred[0]:.2f}")
    print(f"  High : {next_pred[1]:.2f}")
    print(f"  Low  : {next_pred[2]:.2f}")


    # Plotagem dos resultados
    # Cria um gráfico comparando as previsões do Close dos três modelos com os valores reais ao longo do tempo.
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


