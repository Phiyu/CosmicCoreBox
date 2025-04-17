import numpy as np

class particles:
    # a box with width L, and divided to N each side
    def __init__(self, position, velocity, L, N):
        self.x = position
        self.v = velocity
        self.L = L
        self.N = N
        self.n = N*N*N # total number of bins
        self.V = L*L*L # volume

    
    def NGP(self, window_function):
        pass

    def CIC(self, window_function):
        pass