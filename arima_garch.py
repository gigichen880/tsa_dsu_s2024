t1 = "2023-11-01 09:00:00"
t2 = "2023-11-01 10:00:00"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller

from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
register_matplotlib_converters()

from time import time

import itertools

from statsmodels.tools.eval_measures import aic, bic
from sklearn.metrics import mean_squared_error


# Step: Preprocess data
def preprocess_data(path):
    df = pd.read_csv(path, index_col=0)
    df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime
    df.set_index('Date', inplace=True)  # Set 'Date' column as the index
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()  # Drop missing values
    df = df[~df.index.duplicated(keep='first')]
    df = df.resample('T').ffill()  # Resample to ensure consistent minute frequency

    df.index.freq = 'T'
    return df['Log_Returns'].loc[t1:t2]


# ADF Test
def perform_adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])



def find_para(lgret):
    p_range = range(0, 6)
    q_range = range(0, 6)

    # Store results
    results = []

    # Iterate through combinations of p and q
    for p, q in itertools.product(p_range, q_range):
        try:
            model = ARIMA(lgret, order=(p, 0, q))
            model_fit = model.fit()

            # Extract AIC and BIC
            aic_value = model_fit.aic
            # bic_value = model_fit.bic

            # Store results
            results.append((p, q, aic_value))
        except Exception as e:
            print(f"Error with ARIMA({p}, 0, {q}): {e}")

    # Convert results to a DataFrame for easy sorting
    results_df = pd.DataFrame(results, columns=['p', 'q', 'AIC'])

    # Find the best parameters based on AIC and BIC
    best_aic = results_df.loc[results_df['AIC'].idxmin()]
    # best_bic = results_df.loc[results_df['BIC'].idxmin()]

    print("Best parameters based on AIC:")
    print(best_aic['p'], best_aic['q'])

    if not ((best_aic['p'] == 0) & (best_aic['q'] == 0)):
        return int(best_aic['p']), 0, int(best_aic['q'])
    else:
        return 0, 0, 2

    # print("Best parameters based on BIC:")
    # print(best_bic['p'], best_bic['q'])


# # Step 2: Fit ARIMA model
# def fit_arima_model(returns, order=(2, 0, 0)):
#     model = ARIMA(returns, order=order)
#     model_fit = model.fit()
#     return model_fit

# # Step 4: Make predictions for future returns
# def predict_future_returns(model_fit, steps=5):
#     predictions = model_fit.forecast(steps=steps)
#     return predictions

def fit_and_predict(train, test, order):
    predictions = []
    residuals = []
    history = list(train)
    for t in range(len(test)):
        # Predict the next step using the previous values in history (which contains actual values)
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        if t == 0:
            residuals = model_fit.resid

        # One-step forecast
        pred = model_fit.forecast()[0]

        # Save prediction
        predictions.append(pred)

        # Append the true test value (not the prediction)
        history.append(test.iloc[t])
    return predictions, residuals


# Step 5: Visualize the actual returns and forecast
def plot_predictions(returns, predictions, gt):
    plt.figure(figsize=(10, 6))
    plt.plot(returns, label='Actual Log Returns', color='blue')
    plt.plot(predictions, label='Predicted Log Returns', color='red')

    # plt.plot(np.arange(len(returns), len(returns) + len(predictions)), predictions, label='Predicted Log Returns', color='red')
    plt.title("Stock Returns Prediction using ARIMA")
    plt.xlabel("Time")
    plt.ylabel("Log Returns")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(returns, label='Actual Log Returns', color='blue')
    plt.plot(gt, label='True Log Returns', color='green')
    plt.title("Stock Returns Ground Truth")
    plt.xlabel("Time")
    plt.ylabel("Log Returns")
    plt.legend()
    plt.show()


def visualize_and_evaluate(test, predictions, vol):
    # Evaluate the model by calculating RMSE
    rmse = np.sqrt(mean_squared_error(test, predictions))
    print(f"Test RMSE: {rmse}")

    plt.figure(figsize=(10, 6))
    # Plot the results
    plt.plot(test.index, test, label='True Returns')
    plt.plot(test.index, predictions, color='red', linestyle='--', label='Predicted Returns')
    plt.plot(test.index, vol, color='blue', linestyle='dotted', label='Volatility')
    plt.legend()
    plt.title('Stock Return Predictions vs True Returns with Volatility')
    plt.show()

    test_dir = [1 if i >= 0 else -1 for i in list(test)]
    pred_dir = [1 if i >= 0 else -1 for i in list(predictions)]
    corr_dir = [i == j for (i, j) in zip(test_dir, pred_dir)]
    corr_rate = sum(corr_dir) / len(test)
    print(f"Correct Rate of Direction: {corr_rate}")


def get_residual(test, predictions):
    return [t - p for (t, p) in zip(test, predictions)]


def conduct_garch(test_size, resid, p, q):
    rolling_predictions = []
    for i in range(test_size):
        train = resid[:-(test_size - i)]
        model = arch_model(train, p=p, q=q)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=5)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))
    return rolling_predictions

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    path = "./WTI_Minute_Data_2018_2023.csv"
    lgret = preprocess_data(path)

    perform_adf_test(lgret)

    N_test = int(len(lgret) * 0.9)
    train_data = lgret.iloc[:N_test]
    test_data = lgret.iloc[N_test:]

    order = find_para(train_data)

    forecast, resid = fit_and_predict(train_data, test_data, order)

    test = pd.DataFrame(test_data)
    test['forecast'] = forecast

    garch_order = find_para(resid ** 2)
    print(resid)
    print(garch_order)
    warnings.filterwarnings("ignore")
    garch_pred = conduct_garch(len(test_data), resid, garch_order[0], garch_order[2])

    test['volatility'] = garch_pred
    visualize_and_evaluate(test['Log_Returns'], test['forecast'], test['volatility'])