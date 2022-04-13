import matplotlib.pyplot as plt
import numpy as np

# part A
x = np.linspace(-10, 10, 1000) 
f = 2 * np.exp(-x**2) # f(x)
p = np.exp(-abs(x)) # p(x)

plt.figure() # plot f(x) and p(x) ontop of each other 
plt.plot(x, f, label='f(x)')
plt.plot(x, p, label='p(x)')
plt.xlabel('x')
plt.legend()
plt.show()

# same for part B
x = np.linspace(0, np.pi, 1000)
f = 1.5 * np.sin(x)
p = 4 * x/np.pi**2 * (np.pi - x)

plt.figure()
plt.plot(x, f, label='f(x)')
plt.plot(x, p, label='p(x)')
plt.xlabel('x')
plt.legend()
plt.show()