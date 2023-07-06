import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
from scipy.optimize import minimize_scalar
import math


# Define f(x)
def f(x, b):
    return (norm.pdf(x) * 2)/(2 - np.e**(-1/b) - np.e**(-2/b))
# Define g(x)
def g(x, b):
    return (np.e**(-abs(x)/b))/ (b* (2 - np.e**(-1/b)) - np.e**(-2/b))
# Define G-1(x) for the inverval [-1, 0]
def inv_G(x, b):
    return -b * np.log(1 - x*(1 - np.e**(-2/b)))
# Define G-1(x) for the inverval [0, 2]
def inv2_G(x, b):
    return b * np.log(np.e**(-1/b) + x*(1 - np.e**(-1/b)))
# Define h(x) for the interval [-1, 0]
def h1(x, b):
    return (2 * b * (np.e**((-b*x**2 + 2 * x)/(2*b))))/np.sqrt(math.pi * 2)
# Define h(x) for the interval [0, 2]
def h2(x, b):
    return (2 * b * (np.e**((-b*x**2 - 2 * x)/(2*b))))/np.sqrt(math.pi * 2)
# Define the function that gives us M once we got b
def M_func(b):
    return (2 * b * np.e**(1/(2*b**2)))/np.sqrt(math.pi * 2)

# Minimize h in terms of b
res = minimize_scalar(lambda x: M_func(x), bounds=(-1, 2), method='bounded')
# Set b and round it
b = round(res.x)

M = M_func(b)
# Set the ratio explained in theory
ratio = (1-np.e**(-1/b))/(2 - np.e**(-1/b) - np.e**(-2/b))
print("b is : ", b)
print("M is : ", M)

samples = np.zeros(shape=(100000, ))
iteration = 0
while iteration < 100000:
    x = random.uniform(0, 1)
    u = random.uniform(0, 1)
    u2 = random.uniform(0, 1)
# If u2 >= ratio then join [0,2] 
    if u2 >= ratio:
        y = inv_G(x, b)
        if u <= h1(y, b)/M:
            samples[iteration] = y
            iteration += 1
        else:
            continue
# Else join [-1, 0] (more analytical explanation in theory)            
    else:
        y = inv2_G(x, b)
        if u <= h2(y, b)/M:
            samples[iteration] = y
            iteration += 1
        else:
            continue
			
# Generate x and y values for f(x) and g(x)
z = np.linspace(-1, 2, 1000)
f_y = [f(xi, b) for xi in z]
g_y = [g(xi, b)*M for xi in z]

# Plot f
plt.plot(z, f_y, label='f(x)', color = 'b')

# Plot g
plt.plot(z, g_y, label='g(x)', color = 'g')

# Plot the histogram
plt.hist(samples, bins=30, density=True, color = 'r', alpha=0.8, label='Rejection')

plt.title('PDFs')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1,2)
plt.ylim(0,3)
plt.legend()
plt.show()