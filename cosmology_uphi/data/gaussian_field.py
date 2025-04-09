import numpy as np
from scipy.interpolate import CubicSpline
from math import hypot

def PL_fit(x, dataset):
    k = dataset[:,0]
    p = dataset[:,1]
    P = CubicSpline(k, p)
    return P(x)



def norm(L, N, dataset, delta_k: np.ndarray):
    V = L**3
    n = N**3
    kF = 2 * np.pi / L

    # coordinates in Fourier space
    a = np.arange(N)
    a[a >= N//2+1] -= N 
    b = np.arange(N)
    b[b >= N//2+1] -= N
    c = np.arange(N//2 + 1)

    A = a[:, None, None]
    B = b[None, :, None]
    C = c[None, None, :]

    k_mod = kF * np.sqrt(A**2 + B**2 + C**2)

    scale = np.sqrt(PL_fit(k_mod, dataset) * n / V)

    delta_k *= scale
    
    return delta_k


def generate_gf(L, N, dataset, seed = 64):

    np.random.seed(seed)
    grf = np.random.normal(0, 1, size = (N,N,N))
    grf_fft = np.fft.rfftn(grf)

    grf_fft_norm = norm(L, N, dataset, grf_fft)

    return grf_fft_norm