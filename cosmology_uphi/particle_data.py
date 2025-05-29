import numpy as np
from tqdm import tqdm
from .field import field

class particles(field):
    # a box with width L, and divided to N each side
    def __init__(self, position, velocity, field: field):
        super().__init__(field.L, field.N, field.ff)
        self.x = position
        self.v = velocity 


    def DFT(self):
        k_vals = np.fft.fftfreq(self.N, self.h) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
        k = np.stack([kx, ky, kz], axis=-1).reshape(-1, 3) 

        delta_k = np.zeros(k.shape[0], dtype=np.complex128)

        for idx, kvec in tqdm(enumerate(k)):
            phase = np.dot(self.x * self.h, kvec)
            delta_k[idx] = np.sum(np.exp(-1j * phase))
        delta_k_grid = delta_k.reshape(self.N, self.N, self.N)

        return field(self.L, self.N, delta_k_grid)
    
    def NGP(self, deconvolution = True):
        rho = np.zeros((self.N, self.N, self.N))

        for i in range(len(self.x)):
            xi = self.x[i]
            grid = np.round(xi).astype(int) % self.N  # particle -> mesh -> int -> periodic boundary

            ix, iy, iz = grid
            rho[ix, iy, iz] += 1.0

        rho /= np.mean(rho)
        delta = rho - 1    
        fourier_delta = np.fft.rfftn(np.array(delta))

        if deconvolution:
            return field(self.L, self.N, fourier_delta / Window_function(self.L, self.N, 1))
        else:
            return field(self.L, self.N, fourier_delta)




    def CIC(self, deconvolution = True):
        rho = np.zeros((self.N, self.N, self.N))

        for i in range(len(self.x)):
            xi = self.x[i] 
            base = np.floor(xi).astype(int)
            d = xi - base

            for dx in [0, 1]:
                wx = 1 - d[0] if dx == 0 else d[0]
                ix = (base[0] + dx) % self.N
                for dy in [0, 1]:
                    wy = 1 - d[1] if dy == 0 else d[1]
                    iy = (base[1] + dy) % self.N
                    for dz in [0, 1]:
                        wz = 1 - d[2] if dz == 0 else d[2]
                        iz = (base[2] + dz) % self.N

                        weight = wx * wy * wz
                        rho[ix, iy, iz] +=  weight

        rho /= np.mean(rho)
        delta = rho - 1    
        fourier_delta = np.fft.rfftn(np.array(delta))
        if deconvolution:
            return field(self.L, self.N, fourier_delta / Window_function(self.L, self.N, 2))
        else:
            return field(self.L, self.N, fourier_delta)


def Window_function(L, N, p):
    kx = np.fft.fftfreq(N, L/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, L/N) * 2 * np.pi
    kz = np.fft.rfftfreq(N, L/N) * 2 * np.pi  # for rfftn

    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')

    # window function in Fourier space
    W = np.sinc(kx_grid * L / (2 * np.pi * N)) * np.sinc(ky_grid * L / (2 * np.pi * N)) * np.sinc(kz_grid * L / (2 * np.pi * N))

    # Prevent division by zero (W ≈ 0), set cutoff
    W[np.abs(W) < 1e-6] = 1.0

    return W**p