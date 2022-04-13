import numpy as np
import matplotlib.pyplot as plt
from importance_part_a import importance_sampling, p, f


A = 1 / (2 - 2 * np.exp(-10)) # normalisation constant part a
x = np.linspace(-10, 10, 1000)
p_x = np.exp(-abs(x)) # sampling func

s = np.array(importance_sampling(f, p, A, N = 400, x = 15, delta = 15)) # run metropolis
print(s[0], s[1]) # results
accepted = np.array(s[2]) # accepted points
rejected = np.array(s[3]) # rejected points
plt.figure() # plot the accepted points as red, rejected points as blue ontop of the sampling function
plt.plot(x, p_x, label='p(x)') 
plt.plot(accepted[:, 0], accepted[:, 1], 'ro', markersize=3.0, label='accepted points')
plt.plot(rejected[:, 0], rejected[:, 1], 'bo', markersize=1.2, label='rejected points')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
plt.show()