import numpy as np
import matplotlib.pyplot as plt
from box import particles


times = [100, 1000, 10000, 100000, 1000000, 10000000] # timescales 
means = np.zeros((2, len(times)))
std_deviations = np.zeros((2, len(times)))
std_errors = np.zeros((2, len(times)))

for i in range(len(times)):
    results, t = particles(t_final=times[i], 
                           N=500, 
                           n_right=500, 
                           p_right=0.5, 
                           gen=np.random.PCG64(1)) # call box simulation
    means[0][i] = np.mean(results[0]) # mean left side
    means[1][i] = np.mean(results[1]) # mean right side
    std_deviations[0][i] = np.std(results[0]) # std left
    std_deviations[1][i] = np.std(results[1]) # etc
    std_errors[0][i] = std_deviations[0][i] / times[i]
    std_errors[1][i] = std_deviations[1][i] / times[i]

# Plot the mean for each timescale
plt.figure()
plt.plot(times, means[0], label='left side')
plt.plot(times, means[1], '-.', label='right side')
plt.xscale('log')
plt.xlabel('Number of time points')
plt.ylabel('Mean')
plt.legend()
plt.show()

# Plot the standard deviation for each timescale
plt.figure()
plt.plot(times, std_deviations[0], label='left side')
plt.plot(times, std_deviations[1], '-.', label='right side')
plt.xscale('log')
plt.xlabel('Number of time points')
plt.ylabel('Standard Deviation')
plt.legend()
plt.show()

# Plot the standard error for each timescale
plt.figure()
plt.plot(times, std_errors[0], label='left side')
plt.plot(times, std_errors[1], '-.', label='right side')
plt.xscale('log')
plt.xlabel('Number of time points')
plt.ylabel('Standard Error')
plt.legend()
plt.show()