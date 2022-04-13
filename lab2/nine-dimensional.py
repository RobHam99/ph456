import numpy as np
from mc import integrate, convergence # import mc solver
import matplotlib.pyplot as plt

def f(arr):
    """Sampling function

    Args:
        arr (np.ndarray): array with one uniform random number for each dimension

    Returns:
        float: f(x,y,...)
    """
    a = arr[0:3] # (ax, ay, az)
    b = arr[3:6]
    c = arr[6:9]
    return 1 / abs(np.dot((a + b), c))

CONVERGENCE = False
if CONVERGENCE:
    n_arr = np.array([1, 5, 10, 50, 500, 1000, 10000, 100000, 1000000]) # array of N vals
    results = convergence(f, 
                          np.array([ # array of limits
                        [0, 1], 
                        [0, 1], 
                        [0, 1], 
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1]]), 
                          n_arr)
    plt.figure()
    # plot results with error bars
    plt.errorbar(n_arr, results[:, 0], yerr=results[:, 1], fmt='k.', label='monte carlo')
    plt.xscale('log')
    plt.xlabel('N')
    plt.ylabel('I')
    plt.legend()
    plt.show()
else: # print results for one N val
    print(integrate(f,
                    np.array([
                        [0, 1], 
                        [0, 1], 
                        [0, 1], 
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1]]), 
                    100000))