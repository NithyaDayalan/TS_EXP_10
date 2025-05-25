## Exp.no : 10 IMPLEMENTATION OF SARIMA MODEL
### Date : 

### AIM :
To implement SARIMA model using python.

### ALGORITHM :
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions

### PROGRAM :
```
Name : NITHYA D
Register Number : 212223240110
```
#### Import necessary packages
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
```
#### Load the dataset
```
data = pd.read_csv('Gold Price Prediction.csv')
```
#### Automatically detect 'Date' and price column
```
date_col = None
target_col = None

for col in data.columns:
    if 'date' in col.lower():
        date_col = col
for col in data.columns:
    if data[col].dtype in ['float64', 'int64'] and col != date_col:
        target_col = col

if not date_col or not target_col:
    raise ValueError("Could not automatically detect Date or Price column.")
```
#### Convert 'Date' column to datetime format
```
data[date_col] = pd.to_datetime(data[date_col])
data.set_index(date_col, inplace=True)
data.sort_index(inplace=True)
data.dropna(inplace=True)
```
#### Plot the time series
```
plt.plot(data.index, data[target_col])
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.title('Gold Price Time Series')
plt.grid(True)
plt.tight_layout()
plt.show()
```
#### Function to check stationarity using ADF test
```
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data[target_col])
```
#### Plot ACF and PACF
```
plot_acf(data[target_col])
plt.show()

plot_pacf(data[target_col])
plt.show()
```
#### Split the data into training and testing sets
```
train_size = int(len(data) * 0.8)
train, test = data[target_col][:train_size], data[target_col][train_size:]
```
#### Build and fit the SARIMA model
```
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()
```
#### Forecast
```
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
```
#### Calculate error
```
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)
```
#### Plot actual vs predicted
```
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', linestyle='--', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.title('SARIMA Model Predictions - Gold Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
### OUTPUT :
#### Time Series Plot :
![image](https://github.com/user-attachments/assets/3b986672-d3e6-41a6-84b3-3017446d1e9f)

#### After Differencing :
![image](https://github.com/user-attachments/assets/180f9d3a-41ae-4ff5-bc15-ad71818ae8a5)

#### Auto Correlation :
![image](https://github.com/user-attachments/assets/d7fbe699-9072-444d-a615-0641e17f14da)

#### Partial Auto Correlation :
![image](https://github.com/user-attachments/assets/82d22910-76b8-4277-91d5-9f9cd2e99be6)

#### SARIMA Forecast :
![image](https://github.com/user-attachments/assets/b6620a10-a31a-47f0-bd8d-3ee98b9b5e89)

#### RMSE :
![image](https://github.com/user-attachments/assets/eb40f459-0dd0-489f-9c42-a242abc6b159)

### RESULT :
Thus the program run successfully based on the SARIMA model.
