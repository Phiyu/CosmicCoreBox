import numpy as np
from tqdm import tqdm

# local
import cosmology_uphi as cu

if __name__ == "__main__":
    new_realization = 200

    L = 1e3
    N = 128
    
    data = data = np.loadtxt('0.dat')

    for i in tqdm(range(new_realization)):
        np.save(f'new_realization_{i}.npy', cu.gf(L, N, data, 85+i))