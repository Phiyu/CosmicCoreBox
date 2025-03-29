import numpy as np
from scipy.integrate import quad


def E(z, m0, Lambda0, r0):
    # m0, Lambda0, r0 refer to Omega_{m,0}, Omega_{Lambda, 0} and Omega_{radiation, 0} respectively.
    k = 1- m0 - Lambda0 - r0 # The curvature component
    return np.sqrt(Lambda0 + m0 * pow(1+z,3)+ r0 * pow(1+z, 4) + k * pow(1+z, 2))


def integrand(x, m0, Lambda0, r0):
    return (1+x)/pow(E(x, m0, Lambda0, r0),3)

def normalization(m0, Lambda0, r0):
    return E(0, m0, Lambda0, r0) * quad(integrand, 0, np.inf, args=(m0, Lambda0, r0))[0]

def D(z, m0, Lambda0, r0):
    if np.ndim(z) == 1:
        results = []
        for z0 in z:
            result = E(z0, m0, Lambda0, r0) * quad(integrand, z0, np.inf, args=(m0, Lambda0, r0))[0]
            results.append(result)
        return np.array(results)/normalization(m0, Lambda0, r0)
    if np.ndim(z) == 0:
        return E(z, m0, Lambda0, r0) * quad(integrand, z, np.inf, args=(m0, Lambda0, r0))[0]/normalization(m0, Lambda0, r0)
    else:
        raise ValueError("z is not a number or an 1-d array.")




if __name__ == "__main__":
    r0 = 0
    m0 = 0.3
    Lambda0 = 0.7

    print(D(0, m0, Lambda0, r0))
    print(D([0,1,2,3], m0, Lambda0, r0))