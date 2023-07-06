import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def f(x):
    y = np.zeros_like(x)
    mask1 = (x > math.pi/2) & (x <= math.pi)
    mask2 = (x >= 0) & (x <= math.pi/2)
    y[mask1] = 4/math.pi - 4*x[mask1]/math.pi**2
    y[mask2] = 4*x[mask2]/math.pi**2
    return y
	
# MAIN CODE
iteration = 0
samples = np.zeros(shape=(1000, ))
x = 0 # define x

while iteration < 1000:
    
    # Step 1: Generate a uniform value u in (0,1)
    u = random.uniform(0, 1)
    # Check in which interval it belongs
    if u > 1/2:
        x1 = math.pi*((2 + math.sqrt(2-2*u))/2)
        x2 = math.pi*((2 - math.sqrt(2-2*u))/2)
        # One of the x's should be rejected since F is '1-1' in [pi/2, pi]
        if x1 >= math.pi/2 and x1 <= math.pi:
            x = x1
        else:
            x = x2
        
    else:
        x = math.sqrt(u/2)*math.pi

    samples[iteration] = x
    iteration += 1
	
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