import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    X=np.insert(X,0,1,axis=1)
    weights=np.zeros(len(X[0]))
    m=len(X)
    for i in range(steps):
        y_hat=_sigmoid(np.dot(X,weights))
        gradient=np.dot((y-y_hat),X)/m
        weights=weights+ lr*gradient

    return (weights[1:],float(weights[0]))