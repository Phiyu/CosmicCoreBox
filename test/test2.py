import numpy as np

# local
import cosmology_uphi as cu

if __name__ == "__main__":
    z = np.linspace(0, 10, 10000)
    x = [np.log10(1+z), np.log10(1+z), np.log10(1+z)]
    y = [cu.D(z, 1, 0, 0), cu.D(z, 0.7, 0.3, 0), cu.D(z, 0.3, 0.7, 0)]
    color = ['red', 'orange', 'blue']
    label = ["standard CDM", "LCDM", "open CDM"]
    
    cu.fplt(x, y, color, label ,"Linear Growth Factor", r'$\log (1+z)$', r'$D(z)$', "normal", "log")
