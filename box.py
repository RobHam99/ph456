import numpy as np
import matplotlib.pyplot as plt

def particles(t1 = 10000, N = 500, n_right = 500, p_right = 0.5, gen = np.random.PCG64(2022)):
    """Simulate the diffusion of particles starting in 1 side of a box.

    Args:
        t1 (int, optional): number of time steps. Defaults to 10000.
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
    rng = np.random.Generator(gen)
    final_arr = np.zeros((2, t1))
    n_left = N - n_right # number of particles on left side
    for t in range(t1):
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
    return final_arr

p_right = 0.25
results = particles(t1 = 10000, N = 500, n_right = 500, p_right = p_right, gen = np.random.Philox(2022))

plt.figure()
plt.plot(results[0], label='number on left side')
plt.plot(results[1], label='number on right side')
plt.xlabel('time')
plt.ylabel('N particles')
plt.legend()
plt.show()