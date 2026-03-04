import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from pmdarima import auto_arima
#download dos dados
ticker = "BCP.LS"
data = yf.download(ticker, start="2020-01-01", end="2025-12-31")
data = data.resample('W-FRI').last().ffill().dropna()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

#separação entre dados para teste
train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]


# retornos diários
returns = train['Close'].pct_change().dropna()

#auto arima para verificar o p,d,q
auto_model = auto_arima(
    returns,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore"
)
best_order = auto_model.order
print("Melhor ordem ARIMA (p,d,q):", best_order)

#arma com os os parametros estabelecidos com o autoarima
model = ARIMA(returns, order=best_order)
model_fit = model.fit()


#previsão de retornos e recontrução de preços por causa do -1
n_forecast = len(test)
pred_ret = model_fit.forecast(steps=n_forecast)

last_train_price = train['Close'].iloc[-1]
pred_prices = last_train_price * (1 + pred_ret).cumprod()
pred_prices.index = test.index  # alinha com test

#erro médio -> o tamanho dos erros de previsão
mae = mean_absolute_error(test['Close'], pred_prices)
print(f"MAE: {mae:.4f}")

#ROI E SHARPE
test_prices = test['Close'].values
pred_values = pred_prices.values


#calculo de lucro (mas não está a executar ñ sei porquê)
profits = []
for i in range(len(pred_values) - 1):
    today_price  = test_prices[i]
    tomorrow_real = test_prices[i + 1]

    signal = pred_values[i + 1] > today_price  

    if signal:                             
        profit = tomorrow_real - today_price
    else:                                
        profit = today_price - tomorrow_real

    profits.append(profit)
profits = np.array(profits)
roi = profits.sum() / test_prices[0]
sharpe = profits.mean() / (profits.std() + 1e-8)

print(f"ROI:           {roi:.4f}")
print(f"Sharpe Ratio:  {sharpe:.4f}")

#correção para melhorar a previsão 
correct = 0
for i in range(len(pred_values) - 1):
    today_price   = test_prices[i]
    tomorrow_real = test_prices[i + 1]
    tomorrow_pred = pred_values[i + 1]  

    real_up = tomorrow_real > today_price
    pred_up = tomorrow_pred > today_price

    if real_up == pred_up:
        correct += 1

accuracy = correct / len(profits)
print(f"Accuracy direcional: {accuracy:.4f}")

#gráfico só para visualizar os dados reais vs previstos
plt.figure(figsize=(14, 6))
plt.plot(train.index, train['Close'], label="Treino", color="steelblue")
plt.plot(test.index,  test['Close'],  label="Real",   color="green")
plt.plot(test.index,  pred_prices,    label="ARIMA", color="red", linestyle="--")
plt.title(f"ARIMA — {ticker} (Retornos Diários)")
plt.xlabel("Data")
plt.ylabel("Preço de Fecho")
plt.legend()
plt.tight_layout()
plt.show()