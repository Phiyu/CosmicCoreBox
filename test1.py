import numpy as np

# local
import cosmology_uphi as cu


if __name__ == "__main__":
    # load data.dat
    data = np.loadtxt('0.dat')
    L = 1e3
    N = 128

    pm = cu.pm_data(data, L, N)
    print(len(pm.k_vals), len(pm.pm))

    print(pm.calculate_sigma())
    
    gf = pm.generate_filed()

    kF = 2* np.pi/L
    k_grid = np.arange(kF, N*kF/2+kF/10, kF)
    cu.fplt(cu.estimator(L, N, gf)[0], cu.estimator(L, N, gf)[1], "red", "binned correction", "Power Spectrum", "k", "P(k)", "log", "log")