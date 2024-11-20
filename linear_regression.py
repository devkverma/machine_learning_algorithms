import matplotlib.pyplot as plt
import numpy as np
from typing import List

class LinearRegression:

    #class attributes
    intercept_ = 0
    coef_ = 0

    def __init__(self):

        if 'np' not in globals() and 'numpy' not in globals():
            raise ImportError("Missing Dependency: numpy is not imported. Please import numpy as 'np'.")
        pass

    def fit(self, x : List[float], y : List[float]) -> 'LinearRegression':

        x = np.array(x)
        y = np.array(y)

        # Dimension checks
        if x.ndim > 2 or y.ndim != 1:
            raise ValueError("Dimension Mismatch Error: Input should be of 2 dimension atmost and Output should be of 1 dimension only.")
        
        # Length check
        if len(x) != len(y):
            raise ValueError("Length Mismatch: Number of features is not equal to number of targets.")
        
        # Converting 1-D array to 2-D
        if x.ndim == 1:
            x = x.reshape(-1,1)

        # Stacking ones column for the intercept term
        x = np.hstack((np.ones((x.shape[0],1)),x))

        # Compute normal equation
        try: 
            theta = np.linalg.inv(x.T @ x) @ x.T @ y
        except np.linalg.LinAlgError:
            raise ValueError("Singular Matrix: Matrix is not invertible, try reducing multi-collinearity")

        # store intercept and coefficients
        self.intercept_, self.coef_ = theta[0], theta[1:]

        return self
    
    def predict(self, x : List[float]) -> np.ndarray :

        x = np.array(x)

        if x.ndim == 1:
            x = x.reshape(-1,1)
        
        x = np.hstack((np.ones((x.shape[0],1)),x)) # add intercept term

        return x @ np.concatenate(([self.intercept_],self.coef_))

if __name__ == "__main__":
    x = np.array([[1, 2], [2, 1], [3, 3]])  # 3 samples, 2 features
    y = np.array([5, 7, 9])  # Target values

    # Initialize and train the model
    model = LinearRegression()
    model.fit(x, y)

    # Print the learned intercept and coefficients
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {model.coef_}")

    # Make predictions on new data
    x_new = np.array([[4, 5], [5, 6]])
    predictions = model.predict(x_new)
    print(f"Predictions: {predictions}")
    