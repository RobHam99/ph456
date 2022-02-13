import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, chisquare


def generate_(N, generator_):
    """Generate a random sequence of numbers.

    Args:
        N (int): number of points
        seed_ (int): seed value
        generator_ (obj): random generator to use

    Returns:
        np.ndarray: array of random numbers
    """
    rng = np.random.Generator(generator_)
    sequence = rng.random([N])
    return sequence
    
    
def chi_test(arr):
    """Chi square test

    Args:
        arr (np.ndarray): array of random numbers

    Returns:
        tuple: chi squared value, number of bins 
    """
    y, M, _ = plt.hist(arr)
    N = len(arr)
    M = len(M) - 1
    chi2 = 0
    for i in range(M):
        E_i = N / M
        chi2 += (y[i] - E_i)**2 / E_i
    return chi2, M


SEEDS = [1, 2022]
av_chi = np.zeros(len(SEEDS)) # average chi^2 for each of 2 generators
corr_arr_pc = []
corr_arr_ph = []

"""
pc = generate_(20000, np.random.Philox(1))
plt.figure()
plt.hist(pc)
plt.xlabel("Number value")
plt.ylabel("Number of numbers")
plt.show()
"""
for i in range(len(SEEDS)):
    pc = generate_(20000, np.random.PCG64(SEEDS[i]))
    ph = generate_(20000, np.random.Philox(SEEDS[i]))
    av_chi[0] += chi_test(pc)[0] / len(SEEDS)
    av_chi[1] += chi_test(ph)[0] / len(SEEDS)

    corr_arr_pc.append(np.correlate(pc, pc, 'same'))
    corr_arr_ph.append(np.correlate(ph, ph, 'same'))
print("Average chi squared PCG64, Philox: ", av_chi)
print(corr_arr_pc[0])
# auto correlation plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.tight_layout()
ax1.plot(corr_arr_pc[0])
ax1.set_title('Auto-correlation PCG64')
ax1.legend(['Seed = 1'])
ax2.plot(corr_arr_pc[1])
ax2.set_title('Auto-correlation PCG64')
ax2.legend(['Seed = 2022'])
ax3.plot(corr_arr_ph[0])
ax3.set_title('Auto-correlation Philox')
ax3.legend(['Seed = 1'])
ax4.plot(corr_arr_ph[1])
ax4.set_title('Auto-correlation Philox')
ax4.legend(['Seed = 2022'])
plt.show()