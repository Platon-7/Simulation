import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def h(x):
    return 4*x/math.pi**2
    
# Define the inverse CDF of h(x)
def inv_H(u):
    return math.sqrt(u*(math.pi**2)/2)

def f(x):
    y = np.zeros_like(x)
    mask1 = (x > math.pi/2) & (x <= math.pi)
    mask2 = (x >= 0) & (x <= math.pi/2)
    y[mask1] = 4/math.pi - 4*x[mask1]/math.pi**2
    y[mask2] = 4*x[mask2]/math.pi**2
    return y
    
# Transform the sample to get a sample from f(x)
x_transformed = np.zeros(shape=(1000,))
iteration = 0
while iteration < 1000:
    temp = 0
    u = random.uniform(0, 1)
    if u <= 1/2:
        temp = math.sqrt(u/2)*math.pi
    else:
        temp = math.pi - math.sqrt((1-u)/2)*math.pi
        
    x_transformed[iteration] = temp
    iteration += 1

xs = np.linspace(x_transformed.min(), x_transformed.max(), 1000)
ys = f(xs)

# Plot the histogram
plt.hist(x_transformed, bins=30, density=True, alpha=1, label='Histogram')

# Estimate the PDF curve using KDE
kde = gaussian_kde(x_transformed)
z = np.linspace(x_transformed.min(), x_transformed.max(), 1000)
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