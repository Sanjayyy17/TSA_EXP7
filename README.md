# Ex.No: 07  AUTO REGRESSIVE MODEL

### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```
NAME: MANJUSRI KAVYA R
REG.NO: 212224040186
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```
```
data = pd.read_csv("/content/plane_crash.csv", parse_dates=["Date"])
```
```
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])
data = data.set_index('Date')
```
```
monthly_crashes = data.resample('M').size()
monthly_crashes = monthly_crashes.to_frame(name="Crashes")
```
```
result = adfuller(monthly_crashes['Crashes'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```
```
x = int(0.8 * len(monthly_crashes))
train_data = monthly_crashes.iloc[:x]
test_data = monthly_crashes.iloc[x:]
```
```
lag_order = 13
model = AutoReg(train_data['Crashes'], lags=lag_order)
model_fit = model.fit()
```
```
plt.figure(figsize=(10, 6))
plot_acf(monthly_crashes['Crashes'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()
```
```
plt.figure(figsize=(10, 6))
plot_pacf(monthly_crashes['Crashes'], lags=40, alpha=0.05, method='ywm')
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
```
```
predictions = model_fit.predict(
    start=len(train_data),
    end=len(train_data) + len(test_data) - 1
)
```
```
mse = mean_squared_error(test_data['Crashes'], predictions)
print('Mean Squared Error (MSE):', mse)

```
```
plt.figure(figsize=(12, 6))
plt.plot(test_data['Crashes'], label='Test Data - Number of Crashes')
plt.plot(predictions, label='Predictions - Number of Crashes', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Number of Crashes')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid()
plt.show()
```

### OUTPUT:

### ACF:

<img width="625" height="508" alt="image" src="https://github.com/user-attachments/assets/a3f6c860-16bf-417a-bd05-bf6dc382ee79" />

### PACF:

<img width="636" height="503" alt="image" src="https://github.com/user-attachments/assets/9a61776f-aaa5-4c1b-9261-079e917ff336" />


### Mean Squared Error (MSE):

<img width="381" height="28" alt="image" src="https://github.com/user-attachments/assets/00689094-d6cc-4f51-9c56-ce465898e8a1" />


### FINIAL PREDICTION:
<img width="1128" height="612" alt="image" src="https://github.com/user-attachments/assets/467e7b78-0256-4a0d-a581-26689dae37a7" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
