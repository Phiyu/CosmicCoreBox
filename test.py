import numpy as np

# local
from  cosmology_uphi import data


if __name__ == "__main__":
    # load data.dat
    data0 = np.loadtxt('0.dat')

    print(data.sigma8.calculate_sigma(data0))