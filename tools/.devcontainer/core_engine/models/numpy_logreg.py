import numpy as np

class NumpyLogReg:
    def __init__(self):
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    def _std(self, X):
        if self.mu is None:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0) + 1e-8
        return (X - self.mu) / self.sigma

    def fit(self, X, y, lr=0.1, steps=800):
        X = self._std(np.asarray(X, dtype=np.float64))
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0
        for _ in range(steps):
            z = X @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = X.T @ (p - y) / n
            grad_b = (p - y).mean()
            self.w -= lr * grad_w
            self.b -= lr * grad_b
        return self

    def predict_proba(self, X):
        X = self._std(np.asarray(X, dtype=np.float64))
        z = X @ self.w + self.b
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T




