import numpy as np

def AIC(Y, Yhat, k):
    N = len(Y)
    RSS = np.sum((Y - Yhat)**2)
    nu  = N - k
    return 2*k + N*np.log(RSS/nu)

def BIC(Y, Yhat, k):
    N = len(Y)
    RSS = np.sum((Y - Yhat)**2)
    nu  = N - k
    return k*np.log(N) + N*np.log(RSS/nu)

def nmse(Y, Yhat, k=0):
    return  100*np.sum((Y - Yhat)**2) / (np.var(Y)*Y.shape[0])

def rmse(Y, Yhat, k=0):
    return np.sqrt(np.sum((Y - Yhat)**2)/Y.shape[0])