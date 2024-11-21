import numpy as np
from typing import List

class LogisticRegression:

    intercept_ = 0
    coef_ = 0
    mean_ = 0
    std_ = 0

    def __init__(self, learningRate : float = 0.01, iterations : int = 1000, threshold : float = 0.5):
        self.learningRate = learningRate
        self.iterations = iterations
        self.threshold = threshold

    def __sigmoid(self, z : np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def fit(self, x : List[float], y : List[int]) -> 'LogisticRegression':
        
        x = np.array(x)
        y = np.array(y)

        # Dimensional checks
        if x.ndim > 2:
            raise ValueError("Dimension Error: Features can't have dimensions more than 2")
        
        if y.ndim != 1:
            raise ValueError("Dimension Error: Targets can only have 1 dimension")
        
        # Adding dimension
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # normalize inputs
        self.mean_ = np.mean(x, axis = 0)
        self.std_ = np.std(x, axis = 0)
        x = (x - self.mean_) / self.std_

        # Stacking ones to column
        x = np.hstack((np.ones((x.shape[0], 1)), x))

        # initializing coefficients
        self.coef_ = np.zeros(x.shape[1])

        # Gradient descent loop
        for _ in range(self.iterations):
            predictions = self.__sigmoid(np.dot(x, self.coef_))
            error = predictions - y
            gradient = np.dot(x.T, error) / len(y)
            self.coef_ -= self.learningRate * gradient

        self.intercept_, self.coef_ = self.coef_[0], self.coef_[1:]
        return self

    def predict(self, x : List[float]) -> np.ndarray:
        
        x = np.array(x)
        
        if x.ndim == 1:
            x = x.reshape(-1,1)

        # normalize inputs
        x = (x - self.mean_) / self.std_

        # Stacking ones to column
        x = np.hstack((np.ones((x.shape[0], 1)), x))

        if x.shape[1] != len(self.coef_) + 1:  # +1 for intercept
            raise ValueError("Dimension Error: Input dimensions do not match the model's coefficients.")
        
        probabilities = self.__sigmoid(np.dot(x, np.concatenate(([self.intercept_], self.coef_))))

        return (probabilities > self.threshold).astype(int)


if __name__ == "__main__":
    
    x = [90, 85, 75, 60, 50, 30, 10, 100, 95, 61, 63, 65, 67, 69]
    y = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]

    model = LogisticRegression()
    model.fit(x,y)

    print(model.coef_)
    print(model.intercept_)

    x_test = [77, 67, 10]

    print(model.predict(x_test))