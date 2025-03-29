import numpy as np
from scipy.interpolate import CubicSpline

def PL_fit(x, dataset):
    k = dataset[:,0]
    p = dataset[:,1]
    P = CubicSpline(k, p)
    return P(x)

def P_bc(L, N, dataset):
    kF = 2*np.pi/L

    k = list(dataset[:,0])
    p = list(dataset[:,1])
    k = np.array(k)

    k_grid = np.arange(kF, N*kF/2+kF/10, kF) # k value
    sum = np.zeros_like(k_grid, dtype = float)
    count =np.zeros_like(k_grid, dtype = int) # number of mode

    for i in range(k.shape[0]):
        idx = int(k[i]//kF) -1
        if idx <N/2:
            count[idx] += 1
            sum[idx] += p[i]

    return sum/count
