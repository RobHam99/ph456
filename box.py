import numpy as np
import matplotlib.pyplot as plt

def particles(t_final=10000, N=500, n_right=500, p_right=0.5, gen=np.random.PCG64(2022)):
    """Simulate the diffusion of particles starting in 1 side of a box.

    Args:
        t_final (int, optional): number of time steps. Defaults to 10000.
        N (int, optional): number of particles. Defaults to 500.
        n_right (int, optional): number of particles on right side
        at t=0. Defaults to 500.
        p_right (float, optional): probability of particle moving
        left to right. Defaults to 0.5.
        gen (np.random.generator, optional): generator of random numbers.
        Defaults to PCG64.

    Returns:
        tuple: arrays of the timeseries for particles in 
        left and right side of box, respectively
    """
    if n_right > N:
        raise Exception("Number of particles starting on the right side cannot \
                        exceed total number of particles.")
    if n_right < 0 or N < 0:
        raise Exception("Number of particles must be positive.")
    rng = np.random.Generator(gen)
    final_arr = np.zeros((2, t_final))
    t_arr = []
    n_left = N - n_right # number of particles on left side
    for t in range(t_final):
        t_arr.append(t)
        final_arr[0][t] = n_left
        final_arr[1][t] = n_right
        p = n_right / N # probability of move to right
        r = rng.random([2]) # generate 2 random numbers
        if r[0] <= p and r[1] > p_right: # move left
            n_right -= 1
            n_left += 1
        elif r[0] > p and r[1] <= p_right: # move right
            n_left -= 1
            n_right += 1
    return final_arr, t_arr


def standard_error(arr):
    """Calculate standard error of array

    Args:
        arr (np.ndarray): array to find standard error.

    Returns:
        np.ndarray: array of standard errors.
    """
    return np.std(arr) / np.sqrt(len(arr))


if __name__ == '__main__':
    results, times = particles(t_final=10000, 
                               N=500, 
                               n_right=500, 
                               p_right=0.5, 
                               gen=np.random.PCG64(1))

    # plot timeseries of number of particles on each side of box
    plt.figure()
    plt.errorbar(times, results[0], yerr=standard_error(results[0]), label='number on left side')
    plt.errorbar(times, results[1], yerr=standard_error(results[1]), label='number on right side')
    plt.xlabel('time')
    plt.ylabel('N particles')
    plt.legend()
    plt.show()