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

    Returns:
        tuple: arrays of the timeseries for particles in 
        left and right side of box, respectively
    """
    rng = np.random.Generator(gen)
    l_arr = np.zeros(t1)
    r_arr = np.zeros(t1)
    t = 0
    n_left = N - n_right # number of particles on left side
    while t < t1:
        l_arr[t] = n_left
        r_arr[t] = n_right
        p = n_right / N # probability of move to right
        r1 = rng.random([1]) # decide move left or right
        r2 = rng.random([1]) # decide move or not
        print(r1, r2)
        if r1 <= p and r2 > p_right:
            n_right -= 1
            n_left += 1
        elif r1 > p and r2 <= p_right:
            n_left -= 1
            n_right += 1
        t += 1
    return l_arr, r_arr

p_right = 0.5
l, r = particles(t1 = 10000, N = 500, n_right = 500, p_right = p_right, gen = np.random.Philox(2022))


plt.figure()
plt.plot(l, label='number on left side')
plt.plot(r, label='number on right side')
#plt.title(f'Particles in a box, probability of moving right: {p_right*100}%')
plt.xlabel('time')
plt.ylabel('N particles')
plt.legend()
plt.show()