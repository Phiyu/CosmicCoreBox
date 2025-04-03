import numpy as np

# local
import cosmology_uphi as cu


if __name__ == "__main__":
    # load data.dat
    data = np.loadtxt('0.dat')
    L = 1e3
    N = 128

    print(cu.sigma8.calculate_sigma(data))
    
    gf = cu.gf(L, N, data)
    print(gf.shape())

    kF = 2* np.pi/L
    k_grid = np.arange(kF, N*kF/2+kF/10, kF)
    cu.fplt(k_grid, cu.bc(L, N, data), "red", "binned correction", "log", "log")