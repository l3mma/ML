import numpy as np

class MyLinReg:
    def __init__(self, n_iter = 50, learning_rate = 0.1, weights = None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
    
    def fit(self, X,y,verbose = False):
        self.X = X.copy()
        self.y = y.copy()
        self.X = np.hstack([np.ones((len(X), 1)), X])
        self.loss = 0

        for i in range(1, self.n_iter):
            y_pred =  self.X @ self.weights
            self.loss = np.mean((y_pred - self.y) ** 2)
            grad = 2/len(self.X) * (y_pred - self.y) @ self.X
            self.weights -= self.learning_rate * grad
            if verbose and verbose % i == 0:
                print(f'{i} | loss: {self.loss}')
        
    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def get_coef(self):
        return self.weights[1:]