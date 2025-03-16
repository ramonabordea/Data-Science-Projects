import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Load dataset
file_path = "powerconsumption.csv"
df = pd.read_csv(file_path)

# Convert Datetime column to datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Set Datetime as index
df.set_index('Datetime', inplace=True)

# Resampling data to hourly level
power_data = df['PowerConsumption_Zone1'].resample('H').mean()

# Check for stationarity using ADF test
def adf_test(series):
    result = adfuller(series.dropna())
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] <= 0.05:
        print("The data is stationary")
    else:
        print("The data is non-stationary")

print("Checking stationarity of Power Consumption...")
adf_test(power_data)

# Differencing the series if non-stationary
diff_data = power_data.diff().dropna()
print("Checking stationarity after differencing...")
adf_test(diff_data)

# Fit ARIMA model
p, d, q = 2, 0, 2  # These values can be optimized further
darima_model = ARIMA(power_data, order=(p, d, q))
result = darima_model.fit()
print(result.summary())

# Forecasting the next 24 hours
forecast_steps = 24
forecast = result.forecast(steps=forecast_steps)

# Plot actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(power_data[-100:], label='Actual Power Consumption')
plt.plot(pd.date_range(start=power_data.index[-1], periods=forecast_steps+1, freq='H')[1:], forecast, label='ARIMA Forecast', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Power Consumption')
plt.title('Power Consumption Forecast using ARIMA')
plt.legend()
plt.show()
