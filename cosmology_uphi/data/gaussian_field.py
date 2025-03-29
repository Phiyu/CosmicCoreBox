import numpy as np
from scipy.interpolate import CubicSpline

def PL_fit(x, dataset):
    k = dataset[:,0]
    p = dataset[:,1]
    P = CubicSpline(k, p)
    return P(x)


def norm(L, N, dataset, fft: np.ndarray):
    V = L*L*L
    n = N*N*N

    kF = 2*np.pi/L
    x, y, z = fft.shape
    for i in range(x):
        a = i
        if i >= N//2+1:
            a -= N
        a = i*i
        for j in range(y):
            b = j
            if j >= N//2+1:
                b -= N
            b = j*j
            for l in range(z):
                k_mod = kF*np.sqrt(a+b+l*l)
                fft[i][j][l] *= np.sqrt(PL_fit(k_mod, dataset)*n/V)
    return fft


def generate_gf(L, N, dataset, seed = 64):
    V = L*L*L
    n = N*N*N

    np.random.seed(seed)
    grf = np.random.normal(0, 1, size = (N,N,N))
    grf_fft = np.fft.rfftn(grf)

    grf_fft_norm = norm(L, N, dataset, grf_fft)

    return grf_fft_norm