import numpy as np
from mc import integrate
import matplotlib.pyplot as plt

def importance_sampling(f, p, A, N, x, delta):
    """Metropolis importance sampling algorithm

    Args:
        f (function): integrand
        p (function): sampling function
        A (float): norm. constant
        N (int): number of samples
        x (float): intial guess
        delta (float): interval

    Returns:
        float: integral
    """
    accepted = 0
    f_mean = 0
    f_squared = 0
    accepted_arr = np.zeros((N, 2)) # store accepted trials x and p(x)
    rejected_arr = np.zeros((N, 2)) # store rejected trials x_t and p(x_t)
    for i in range(N):
        delta_i = np.random.uniform(-delta, delta, 1)
        x_t = x + delta_i # new trial
        w = p(x_t) / p(x) 
        if w >= 1: # accept trial
            x = x_t
            accepted += 1
            accepted_arr[i] = np.array([float(x), float(p(x))])
        elif w < 1:
            r = np.random.uniform(0, 1, 1)
            if r <= w: # also accept trial
                x = x_t
                accepted += 1
                accepted_arr[i] = np.array([float(x), float(p(x))])
            else: # reject trial
                rejected_arr[i] = np.array([float(x_t), float(p(x_t))])
        f_val = f(x)
        p_val = p(x)
        f_mean += 1 / N * f_val / (A * p_val) # <f>
        f_squared += 1 / N * (f_val / (A * p_val))**2 # <f^2>
        
    error = np.sqrt((f_squared - f_mean*f_mean)) / np.sqrt(N) # standard error
    print("Percentage accepted: ", accepted*100/N) # print percentage of accepted trials
    accepted_arr = accepted_arr[~np.all(accepted_arr == 0, axis=1)] # remove zero elements
    rejected_arr = rejected_arr[~np.all(rejected_arr == 0, axis=1)] # '                  '
    return f_mean, error, accepted_arr, rejected_arr
    
def p(x):
    """Sampling function

    Args:
        x (float): random generated number

    Returns:
        float: value from the sampling function
    """
    return np.exp(-abs(x))    

def f(x):
    """Integrand

    Args:
        x (float): random generated number

    Returns:
        float: f(x)
    """
    return 2 * np.exp(-x**2)

if __name__ == '__main__':
    A = 1 / (2 - 2 * np.exp(-10)) # normalisation constant
    sample_array = np.array([1, 10, 100, 1000, 10000, 100000]) # N vals
    importance_array = np.zeros((2, len(sample_array))) # metropolis results, errors
    uniform_array = np.zeros((2, len(sample_array))) # mc results, errors
    
    for i in range(len(sample_array)): # calculate metropolis and mc for each N
        imp = importance_sampling(f, p, A, sample_array[i], 0, 3.1)
        uni = integrate(f, np.array([[-10, 10]]), sample_array[i])
        importance_array[:, i] = imp[0:2]
        uniform_array[:, i] = uni[0:2]
    
    # plot convergence of metropolis and monte carlo, and the analytical sol
    plt.figure()
    plt.errorbar(sample_array, importance_array[0], yerr=importance_array[1], fmt='k.', label='Metropolis')
    plt.errorbar(sample_array, uniform_array[0], yerr=uniform_array[1], fmt='r.', label='Monte Carlo')
    plt.plot(sample_array, np.full(len(sample_array), 3.5449), 'g-', label='analytical solution')
    plt.xlabel('N')
    plt.ylabel('I')
    plt.xscale('log')
    plt.legend()
    plt.show()
    # print results of metropolis with error for N = 50000
    print(importance_sampling(f, p, A, N = 50000, x = 0, delta = 3.1)[0:2])
