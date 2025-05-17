import numpy as np

# local
import cosmology_uphi as cu


if __name__ == "__main__":
    # load data.dat
    data = np.loadtxt('0.dat')
    L = 1e3
    N = 128

    pm = cu.pm_data(data)
    print(len(pm.k_vals), len(pm.pm))

    print(pm.calculate_sigma())
    
    gf = pm.generate_filed(L, N)

    kF = 2* np.pi/L
    k_grid = np.arange(kF, N*kF/2+kF/10, kF)
    cu.fplt(gf.estimator()[0], gf.estimator()[1], "red", "binned correction", "Power Spectrum ", r'Wave Number $k\,\mathrm{h/Mpc}$', r'Power Spectrum $P(k) \, \mathrm{(Mpc/h)}^3$', "log", "log")