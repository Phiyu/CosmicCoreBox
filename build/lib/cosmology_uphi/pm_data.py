import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid

class pm_data:
    def __init__(self, dataset):
        self.k_vals = dataset[:,0]
        self.pm = dataset[:,1]

    def PL_fit(self, x):
        P = CubicSpline(self.k_vals, self.pm)
        return P(x)
    
    def BinnedCorrection(self):
        kF = 2*np.pi/self.L

        k_grid = np.arange(kF, self.N*kF/2+kF/10, kF) # k value
        sum = np.zeros_like(k_grid, dtype = float)
        count =np.zeros_like(k_grid, dtype = int) # number of mode

        for i in range(self.k_vals.shape[0]):
            idx = int(self.k_vals[i]//kF) -1
            if idx < self.N/2:
                count[idx] += 1
                sum[idx] += self.pm[i]

        return sum/count


    def calculate_sigma(self, R = 8):
        k = self.k_vals
        p = self.pm
        Delta2 = k*k*k*p/2/np.pi/np.pi
        integral = Delta2 * W_FT(k, R) *W_FT(k, R) / k
        res = cumulative_trapezoid(integral, k)[-1] # return the last result of the integral
        return np.sqrt(res)


    def generate_filed(self, L, N, seed = 64):
        np.random.seed(seed)
        grf = np.random.normal(0, 1, size = (N,N,N))
        grf_fft = np.fft.rfftn(grf)

        grf_fft_norm = norm(self, L, N, grf_fft)

        from .field import field
        return field(L, N, grf_fft_norm)

def norm(data: pm_data, L, N, delta_k: np.ndarray):
    kF = 2 * np.pi / L
    n = N*N*N
    V = L*L*L

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

    scale = np.sqrt(data.PL_fit(k_mod) * n / V)

    delta_k *= scale
    
    return delta_k

def W_FT(k, R):
    x = k*R
    return 3*(np.sin(x)-x*np.cos(x))/(pow(x,3))
