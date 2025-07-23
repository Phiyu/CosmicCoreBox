import numpy as np
from math import hypot

class field:
    def __init__(self, L, N, fourier_field_array = None):
        self.L = L
        self.N = N
        self.kF = 2*np.pi/L # fundamental k
        self.h = L/N 
        self.kN = np.pi/(L/N) # Nyquist Frequency
        self.n = N*N*N # total number of bins
        self.V = L*L*L # volume of the box (field)
        self.ff = fourier_field_array

    def estimator(self): 
        k_grid = np.arange(self.kF, self.N*self.kF/2+self.kF/10, self.kF) # k value
        sum = np.zeros_like(k_grid, dtype = float)
        P = np.zeros_like(k_grid, dtype = float) # ps
        count =np.zeros_like(k_grid, dtype = int) # number of mode

        s = int(k_grid.shape[0])

        x, y, z = self.ff.shape # type: ignore
        for i in range(x):
            a = i
            if i >= self.N//2+1:
                a -= self.N
            for j in range(y):
                b = j
                if j >= self.N//2+1:
                    b -= self.N
                for l in range(z):
                    c = l
                    if l >= self.N//2+1:
                        c -= self.N
                    k_mod = self.kF * hypot(a, b, c) 
                    idx = int(k_mod//self.kF) -1
                    if idx <= s-1:
                        count[idx] += 1
                        sum[idx] += abs(self.ff[i][j][l])**2 # type: ignore
                    
        P = sum*self.V/self.n/self.n/count

        return [k_grid, P]
    
    from .ps_data import ps_data
    def BinnedCorrection(self, pm: ps_data):
        kF = 2*np.pi/self.L

        k_grid = np.arange(kF, self.N*kF/2+kF/10, kF) # k value
        sum = np.zeros_like(k_grid, dtype = float)
        count =np.zeros_like(k_grid, dtype = int) # number of mode

        for i in range(pm.k_vals.shape[0]):
            idx = int(pm.k_vals[i]//kF) -1
            if idx < self.N/2:
                count[idx] += 1
                sum[idx] += pm.pm[i]

        dataset = [pm.k_vals, sum/count]

        from .ps_data import ps_data
        return ps_data(dataset)