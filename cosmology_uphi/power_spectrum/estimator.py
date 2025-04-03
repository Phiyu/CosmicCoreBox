import numpy as np

def estimator(L, N, ff: np.ndarray):
    assert ff.shape == (N,N,N//2+1)
    # ff is a Fourier mode, array.shape = (N, N, N/2+1)
    
    kF = 2*np.pi/L
    V = L*L*L
    n = N*N*N
    
    k_grid = np.arange(kF, N*kF/2+kF/10, kF) # k value
    sum = np.zeros_like(k_grid, dtype = float)
    P = np.zeros_like(k_grid, dtype = float) # ps
    count =np.zeros_like(k_grid, dtype = int) # number of mode

    s = int(k_grid.shape[0])

    x, y, z = ff.shape
    for i in range(x):
        a = i
        if i >= N//2+1:
            a -= N
        a = i*i
        for j in range(y):
            b = j
            if j >= N//2+1:
                b -= N
            b = j*j
            for l in range(z):
                k_mod = kF*np.sqrt(a+b+l*l)
                idx = int(k_mod//kF) -1
                if idx <= s-1:
                    count[idx] += 1
                    sum[idx] += abs(ff[i][j][l])**2
                
    P = sum*V/n/n/count

    return [k_grid, P, count]