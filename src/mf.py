import numpy as np
from numba import njit
import time

@njit
def sgd_update(P, Q, data, lr, reg):
    """JIT-compiled inner loop for SGD: heavily optimizes MF training"""
    np.random.shuffle(data)
    for row in range(data.shape[0]):
        u = int(data[row, 0])
        i = int(data[row, 1])
        r = data[row, 2]
        
        pred = np.dot(P[u], Q[i])
        err = r - pred

        # Update P and Q iteratively
        P_u = P[u].copy() 
        Q_i = Q[i].copy()

        grad_p = err * Q_i - reg * P_u
        grad_q = err * P_u - reg * Q_i

        # gradient clipping for numerical stability
        grad_p = np.clip(grad_p, -5.0, 5.0)
        grad_q = np.clip(grad_q, -5.0, 5.0)

        P[u] += lr * grad_p
        Q[i] += lr * grad_q
        
    return P, Q

class MatrixFactorization:
    def __init__(self, n_users, n_items, k=10, learning_rate=0.01, reg_param=0.02, epochs=20):
        self.k = int(k)
        self.lr = learning_rate
        self.reg = reg_param
        self.epochs = int(epochs)
        
        self.P = np.random.normal(0, 0.1, (n_users, self.k))
        self.Q = np.random.normal(0, 0.1, (n_items, self.k))

    def train(self, data):
        # Convert type to generic float64 for compatibility with Numba arrays
        data = np.array(data, dtype=np.float64)
        for epoch in range(self.epochs):
            self.P, self.Q = sgd_update(self.P, self.Q, data, self.lr, self.reg)
    
    def predict(self, user_id, item_id):
        return np.dot(self.P[user_id], self.Q[item_id])

