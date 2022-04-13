import numpy as np
import matplotlib.pyplot as plt


def integrate(f, limits, N):
    """Integrate a function using monte carlo sampling.

    Args:
        f (object): function to integrate
        limits (np.ndarray): array of limits
        N (int): number of points to sample

    Returns:
        float: value of integral
    """
    M = len(limits) # number of dimensions
    f_mean = 0
    f_squared = 0
    for i in range(N):
        random_numbers = np.zeros(M)
        for j in range(M): # generate 1 random number for each dimension
            random_numbers[j] = np.random.uniform(limits[j][0], limits[j][1], 1)
        f_val = f(random_numbers) # calculate f(x)
        f_squared += f_val * f_val / N # <f^2>
        f_mean += f_val / N # <f>
    
    error = np.sqrt((f_squared - f_mean*f_mean) / N) / np.sqrt(N) # standard error

    for k in range(M):
        f_mean *= limits[k][1] - limits[k][0] # (b-a) * <d-c> * ... * <f> 
    return f_mean, error


def convergence(f, limits, n_arr):
    """Return MC results for an array containing different numbers of samples.

    Args:
        f (func): sampling function
        limits (np.ndarray): 2d array containing limits for each dimension
        n_arr (np.ndarray): array of sample sizes

    Returns:
        np.ndarray: array containing the results with errors for each sample 
        size.
    """
    conv_arr = np.zeros((len(n_arr), 2)) # results & errors for each N
    for i in range(len(n_arr)):
        conv_arr[i] = integrate(f, limits, n_arr[i]) # calculate results for N_i
    return conv_arr
    
if __name__ == "__main__": # don't run the stuff below if mc function is imported into another file
    def f(arr):
        """Sampling function

        Args:
            arr (np.ndarray): array with one uniform random number for each dimension

        Returns:
            float: f(x,y,...)
        """
        return (arr[0] * arr[1] + arr[0]) # arr[0] = x, arr[1] = y .....
    
    
    CONVERGENCE = False 
    if CONVERGENCE:
        n_arr = np.array([1, 10, 100, 1000, 10000, 100000]) # plot convergence for multiple N
        results = convergence(f, np.array([[0, 1], [0, 1]]), n_arr)
        
        plt.figure()
        # plot results with error bars
        plt.errorbar(n_arr, results[:, 0], yerr=results[:, 1], fmt='k.', label='monte carlo')
        # plot analytical sol
        plt.plot(n_arr, np.full(len(n_arr), 0.75), label='analytical')
        plt.xscale('log')
        plt.xlabel('N')
        plt.ylabel('I')
        plt.legend()
        plt.show()
    else:
        print(integrate(f, np.array([[0, 1]]), 100000)) # print results for one N
