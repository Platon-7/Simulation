import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar

def h(x):
    return 4*x/math.pi**2

def f(x):
    y = np.zeros_like(x)
    mask1 = (x > math.pi/2) & (x <= math.pi)
    mask2 = (x >= 0) & (x <= math.pi/2)
    y[mask1] = 4/math.pi - 4*x[mask1]/math.pi**2
    y[mask2] = 4*x[mask2]/math.pi**2
    return y

def g(x):
    return math.sin(x)
    

res = minimize_scalar(lambda x: -(h(x)/g(x)), bounds=(0, math.pi/2), method='bounded')
supremum = -res.fun

iteration = 0
samples = np.zeros(shape=(1000, ))

while iteration < 1000:
    # Step 1: Generate a Y from g
    x = random.uniform(0, 1)

    # Step 2: Generate u from U(0, 1)
    u = random.uniform(0, 1)
    # Step 3: Check the interval and reject if necessary
    if u <= 1/2:
        y = np.arccos(1 - x)
        if u <= (4*y)/(math.sin(y)*supremum*math.pi**2):
            samples[iteration] = y
            iteration += 1
        else:
            continue
    else:
        y = math.pi - np.arccos(x)
        if u <= (4*y)/(math.sin(y)*supremum*math.pi**2):
            samples[iteration] = y 
            iteration += 1
        else:
            continue
            
# Plot the histogram
plt.hist(samples, bins=30, density=True, alpha=1, label='Histogram')

# Estimate the PDF curve using KDE
kde = gaussian_kde(samples)
z = np.linspace(samples.min(), samples.max(), 1000)
estimated_pdf = kde(z)
plt.plot(z, estimated_pdf, 'r-', label='PDF')

# Plot the true PDF curve
real_pdf = [f(xi) for xi in z]
plt.plot(z, real_pdf, 'y--', label='True PDF')

# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('PDF')
plt.legend()

# Show the plot
plt.show()