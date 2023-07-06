import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar

def f(x):
    y = np.zeros_like(x)
    mask1 = (x > math.pi/2) & (x <= math.pi)
    mask2 = (x >= 0) & (x <= math.pi/2)
    y[mask1] = 4/math.pi - 4*x[mask1]/math.pi**2
    y[mask2] = 4*x[mask2]/math.pi**2
    return y

def g(x):
    return math.sin(x)/2
    

res_1= minimize_scalar(lambda x: -(f(x)/g(x)), bounds=(0, math.pi/2), method='bounded')
supremum_1 = -res_1.fun

res_2 = minimize_scalar(lambda x: -(f(x)/g(x)), bounds=(math.pi/2, math.pi), method='bounded')
supremum_2 = -res_2.fun

iteration = 0
samples = np.zeros(shape=(1000, ))

while iteration < 1000:
    # Step 1: Generate a Y from g
    x = random.uniform(0, 1)
    y = np.arccos(1 - 2*x)

    # Step 2: Generate u from U(0, 1)
    u = random.uniform(0, 1)
    u2 = random.uniform(0, 1)
    # Step 3: Check the interval and reject if necessary
    if u2 <= 1/2:
        if u <= (8*y)/(math.sin(y)*supremum_1*math.pi**2):
            samples[iteration] = y
            iteration += 1
        else:
            continue
    else:
        if u <= (8*math.pi - 8*y)/(math.sin(y)*supremum_2*math.pi**2):
            samples[iteration] = y
            iteration += 1
        else:
            continue
            
# Plot the histogram
plt.hist(samples, bins=30, density=True, alpha=0.8, label='Histogram')

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