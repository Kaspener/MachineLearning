import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np

def impute(dataset):
    imputer = SimpleImputer(strategy="mean")
    return imputer.fit_transform(dataset)

data = pd.read_csv('winequalityN.csv')

for column in data.columns[1:]:
    data[column] = pd.to_numeric(data[column], errors='coerce')

red_wine = data[data['type'] == 'red']
white_wine = data[data['type'] == 'white']
all_wine = data

def build_and_evaluate_model(wine_data):
    X = wine_data.iloc[:, 1:-1]
    y = wine_data.iloc[:, -1]
    
    X = impute(X)
    
    X_normalized = normalize(X, axis=0)
    
    mse_list = []
    
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=np.random.randint(1000))
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)

        print(f"Среднеквадратичная ошибка составляет: {mse}")
    
    return np.mean(mse_list)

mse_red = build_and_evaluate_model(red_wine)
mse_white = build_and_evaluate_model(white_wine)
mse_all = build_and_evaluate_model(all_wine)

print()
print(f"Средняя ошибка для красного вина: {mse_red}")
print(f"Средняя ошибка для белого вина: {mse_white}")
print(f"Средняя ошибка для всех вин: {mse_all}")