import numpy as np
from scipy.integrate import quad

class universe:
    def __init__(self,Omega_m0, Omega_r0, Omega_L0):
        self.m0 = Omega_m0
        self.r0 = Omega_r0
        self.L0 = Omega_L0
        self.k0 = 1 - Omega_L0 - Omega_m0 - Omega_r0 # The curvature component

    def E(self, z):
        return np.sqrt(self.L0 + self.m0 * pow(1+z,3)+ self.r0 * pow(1+z, 4) + self.k * pow(1+z, 2))


    def integrand(self, x):
        return (1+x)/pow(self.E(x),3)
    
    def linear_growth_factor(self, z):
        normalization = self.E(0) * quad(self.integrand, 0, np.inf, args=(self.m0, self.L0, self.r0))[0]

        if np.ndim(z) == 1:
            results = []
            for z0 in z:
                result = self.E(z0) * quad(self.integrand, z0, np.inf, args=(self.m0, self.L0, self.r0))[0]
                results.append(result)
            return np.array(results)/normalization
        if np.ndim(z) == 0:
            return self.E(z) * quad(self.integrand, z, np.inf, args=(self.m0, self.L0, self.r0))[0]/normalization
        else:
            raise ValueError("z is not a number or an 1-d array.")
