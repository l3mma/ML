import numpy as np
import random

class MyLineReg:
    def __init__(self, n_iter=50, learning_rate=0.1, weights=None, metric=None, l1_coef=0.0, l2_coef=0.0, reg=None, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.metric_value = 0.0
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.reg = reg
        self.sgd_sample = sgd_sample
        self.random_state = random_state
    
    def log(self, loss, metric, verbose):
        if verbose and i % verbose == 0:
                print(f'{i} | loss: {loss:.2f}', end='')
                if self.metric:
                    print(f' | {self.metric}: {self.metric_value:.2f}')
                else:
                    print()

    
    def _calc_metric(self, y_true, y_pred):
        if self.metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif self.metric == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.metric == 'rmse':
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif self.metric == 'mape':
            epsilon = 1e-15
            return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        elif self.metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-15))
        else:
            return 0.0
        
    def compute_grad(self, X_batch, y_pred_batch, y_batch):
        grad_basic = 2 * X_batch.T @ (y_pred_batch - y_batch) / len(X_batch)
        
        if self.reg == "l1":
            return grad_basic + self.l1_coef * np.sign(self.weights)
        elif self.reg == "l2":
            return grad_basic + self.l2_coef * 2 * self.weights
        elif self.reg == "elasticnet":
            return grad_basic + self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
        else:
            return grad_basic
    
    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)
        
        X = np.array(X)
        y = np.array(y)
        self.X = np.hstack([np.ones((len(X), 1)), X])
        self.y = y
        self.weights = np.ones(self.X.shape[1])

        if self.sgd_sample is None:
            sample_size = len(self.X)
        elif isinstance(self.sgd_sample, float):
            sample_size = int(len(self.X) * self.sgd_sample)
        else:
            sample_size = self.sgd_sample

        for i in range(1, self.n_iter + 1):
            if callable(self.learning_rate):
                current_lr = self.learning_rate(i)
            else:
                current_lr = self.learning_rate
            
            if self.sgd_sample is not None:
                sample_rows_idx = random.sample(range(len(self.X)), sample_size)
                X_batch = self.X[sample_rows_idx]
                y_batch = self.y[sample_rows_idx]
            else:
                X_batch = self.X
                y_batch = self.y
            
            y_pred_batch = X_batch @ self.weights
            grad = self.compute_grad(X_batch, y_pred_batch, y_batch)
            self.weights -= current_lr * grad
            
            y_pred = self.X @ self.weights
            if self.reg == "l1":
                self.loss = np.mean((y_pred - self.y) ** 2) + self.l1_coef * sum(np.abs(self.weights[1:]))
            elif self.reg == "l2": 
                self.loss = np.mean((y_pred - self.y) ** 2) + self.l2_coef * sum(self.weights[1:] ** 2) 
            elif self.reg == "elasticnet":
                self.loss = np.mean((y_pred - self.y) ** 2) + self.l1_coef * sum(np.abs(self.weights[1:])) + self.l2_coef * sum(self.weights[1:] ** 2)
            else:
                self.loss = np.mean((y_pred - self.y) ** 2)

            if self.metric:
                self.metric_value = self._calc_metric(self.y, y_pred)

            self.log(self.loss, self.metric, i, verbose)

    def predict(self, X):
        X = np.array(X)
        X = np.hstack([np.ones((len(X), 1)), X])
        return X @ self.weights
        
    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def get_coef(self):
        return self.weights[1:]
    
    def get_best_score(self):
        return self.metric_value
