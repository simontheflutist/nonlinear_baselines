import numpy as np
from scipy.linalg import solve_triangular


def stable_least_squares(X, y):
    Q, R = np.linalg.qr(X)
    return solve_triangular(R, Q.T.dot(y))
