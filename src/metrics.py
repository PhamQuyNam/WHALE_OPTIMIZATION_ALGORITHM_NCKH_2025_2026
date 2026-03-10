import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def rmse(model, data):
    se = 0
    for u, i, r in data:
        u, i = int(u), int(i)
        pred = model.P[u] @ model.Q[i]
        se += (r - pred)**2
    return np.sqrt(se / len(data))

