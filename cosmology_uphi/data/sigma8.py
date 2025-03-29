import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# the window function in Fourier Space
def W_FT(k, R):
    x = k*R
    return 3*(np.sin(x)-x*np.cos(x))/(pow(x,3))

def calculate_sigma(dataset, R = 8):
    k = dataset[:,0]
    p = dataset[:,1]
    Delta2 = k*k*k*p/2/np.pi/np.pi
    integral = Delta2 * W_FT(k, R) *W_FT(k, R) / k
    res = cumulative_trapezoid(integral, k)[-1] # return the last result of the integral
    return np.sqrt(res)


if __name__ == "__main__":

    # load data.dat
    data0 = np.loadtxt('0.dat')

    calculate_sigma(data0)
