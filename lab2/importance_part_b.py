import numpy as np
from mc import integrate
from importance_part_a import importance_sampling
import matplotlib.pyplot as plt


def p(x):
    """Sampling function.

    Args:
        x (float): random number.

    Returns:
        float: value from sampling function.
    """
    return 4 * x * (np.pi - x) / np.pi**2

def f(x):
    """Integrand.   

    Args:
        x (float): random number.

    Returns:
        float: value of integrand for random number.
    """
    return 1.5 * np.sin(x)

A = 3/(2*np.pi)
sample_array = np.array([1, 10, 100, 1000, 10000, 100000])
importance_array = np.zeros(len(sample_array))
uniform_array = np.zeros(len(sample_array))
error_array = np.zeros((2, len(sample_array)))
for i in range(len(sample_array)):
    imp = importance_sampling(f, p, A, sample_array[i], 1.5, 2.3)
    uni = integrate(f, np.array([[0, np.pi]]), sample_array[i])
    importance_array[i] = imp[0]
    uniform_array[i] = uni[0]
    error_array[0][i] = imp[1]
    error_array[1][i] = uni[1]
# plot the sampling function to guess delta/x0
plt.figure()
plt.errorbar(sample_array, importance_array, yerr=error_array[0], fmt='k.', label='Metropolis')
plt.errorbar(sample_array, uniform_array, yerr=error_array[1], fmt='r.', label='Monte Carlo')
plt.plot(sample_array, np.full(len(sample_array), 3), 'g-', label='analytical solution')
plt.xlabel('N')
plt.ylabel('I')
plt.xscale('log')
plt.legend()
plt.show()
print(importance_sampling(f, p, A, 50000, 1.5, 2.3)[0:2])