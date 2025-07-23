import numpy as np
from tqdm import tqdm
from .field import field
# from .dft_c import dft_core

class particles(field):
    # a box with width L, and divided to N each side
    def __init__(self, position, velocity, field: field):
        super().__init__(field.L, field.N, field.ff)
        self.x = position / self.h # dimensionless
        self.v = velocity 
        self.np = position.shape[0]

    
    def DFT(self):
        kF = 2.0 * np.pi / self.L
        k_indices = np.fft.fftfreq(self.N, self.h) * 2 * np.pi
        
        kx, ky, kz = np.meshgrid(k_indices, k_indices, k_indices, indexing='ij')
        delta_k_grid = np.zeros((self.N, self.N, self.N), dtype=np.complex128)

        for p in tqdm(range(self.np)):
            k_dot_x = (kx * self.x[p, 0] + ky * self.x[p, 1] + kz * self.x[p, 2])
            delta_k_grid += np.exp(-1j * k_dot_x) * self.V / self.np

        delta_k_grid[0,0,0] -= kF**3
        delta_k_grid *=  (self.N**3) / self.V
        
        return field(self.L, self.N, delta_k_grid)
    
    def NGP(self, deconvolution = True, interlacing = True):
        rho = np.zeros((self.N, self.N, self.N))

        for i in range(self.np):
            xi = self.x[i]
            grid = np.round(xi).astype(int) % self.N  # particle -> mesh -> int -> periodic boundary

            ix, iy, iz = grid
            rho[ix, iy, iz] += 1.0

        delta = rho / np.mean(rho) - 1    
        delta_k = np.fft.rfftn(np.array(delta))


        if interlacing:
            shift = np.ones_like(self.x) / 2
            position_shifted = (self.x + shift) % self.L
            rho_shifted = np.zeros((self.N, self.N, self.N))

            for i in range(self.np):
                xi = position_shifted[i]
                grid = np.round(xi).astype(int) % self.N  # particle -> mesh -> int -> periodic boundary

                ix, iy, iz = grid
                rho_shifted[ix, iy, iz] += 1.0

            delta_shifted = rho_shifted / np.mean(rho_shifted) - 1    
            delta_shifted_k = np.fft.rfftn(np.array(delta_shifted))
            return field(self.L, self.N, (0.5 * (delta_k + delta_shifted_k)) / Window_function(self.L, self.N, 1))
        if deconvolution:
            return field(self.L, self.N, delta_k / Window_function(self.L, self.N, 1)) 
        else:
            return field(self.L, self.N, delta_k)




    def CIC(self, deconvolution = True, interlacing = True):
        rho = np.zeros((self.N, self.N, self.N))

        for i in range(self.np):
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

        n = self.np / self.V

        delta = rho / np.mean(rho) - 1    
        delta_k = np.fft.rfftn(np.array(delta))

        if interlacing:
            shift = np.ones_like(self.x) / 2
            position_shifted = (self.x + shift) % self.L
            rho_shifted = np.zeros((self.N, self.N, self.N))

            for i in range(self.np):
                xi = position_shifted[i] 
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
                            rho_shifted[ix, iy, iz] +=  weight


            delta_shifted = rho_shifted / np.mean(rho_shifted) - 1    
            delta_shifted_k = np.fft.rfftn(np.array(delta_shifted))
            return field(self.L, self.N, (0.5 * (delta_k + delta_shifted_k)) / Window_function(self.L, self.N, 1))
        if deconvolution:
            return field(self.L, self.N, delta_k / Window_function(self.L, self.N, 2))
        else:
            return field(self.L, self.N, delta_k)


    # def DFT_power_spectrum(self, k_vals, n1 = 100, n2 = 200):

    #     P = np.zeros_like(k_vals)
    #     for idx, k in tqdm(enumerate(k_vals)):
    #         theta = np.linspace(0, np.pi, n1)
    #         phi = np.linspace(0, 2*np.pi, n2)
    #         theta, phi = np.meshgrid(theta, phi, indexing='ij')

    #         x = k * np.sin(theta) * np.cos(phi)
    #         y = k * np.sin(theta) * np.sin(phi)
    #         z = k * np.cos(theta)

    #         kvec = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    #         n = self.np/self.V
    #         kF = self.kF
    #         sum = 0

    #         for i, _ in enumerate(kvec):
    #             phase = np.dot(self.x * self.h, kvec[i])
    #             sum += abs(1 / n * np.sum(np.exp(-1j * phase)) - kronecker(k) / pow(kF, 3))**2
            
    #         P[idx] = (kF**3)*(sum / n1 / n2)

    #     return P

def Window_function(L, N, p):
    kx = np.fft.fftfreq(N, L/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, L/N) * 2 * np.pi
    kz = np.fft.rfftfreq(N, L/N) * 2 * np.pi  # for rfftn

    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')

    W = np.sinc(kx_grid * L / (2 * np.pi * N)) * np.sinc(ky_grid * L / (2 * np.pi * N)) * np.sinc(kz_grid * L / (2 * np.pi  * N))
    # np.sinc = sin(pi x)/(pi x)

    # Prevent division by zero (W ≈ 0), set cutoff
    W[np.abs(W) < 1e-8] = 1.0

    return W**p

def kronecker(k):
    if k:
        return 0
    else:
        return 1