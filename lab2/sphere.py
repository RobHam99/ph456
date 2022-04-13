import numpy as np
from mc import integrate, convergence
import matplotlib.pyplot as plt

def sphere(arr):
    """Sampling function

    Args:
        arr (np.ndarray): array with one uniform random number for each dimension

    Returns:
        float: f(x,y,...)
    """
    r = 0 # radius
    for i in arr:
        r += i**2
    r = np.sqrt(r)
    if r > 2:
        return 0
    elif r <= 2:
        return 1
    
# print results for one sample size
print(integrate(sphere, np.array([[-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2]]), 5000000))