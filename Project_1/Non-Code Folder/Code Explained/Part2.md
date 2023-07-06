These chunks contain code from the whole 2nd exercise

**METHOD OF INVERSION**

# Define h(x)
def h(x):
    return 4*x/math.pi**2
    
# Define the inverse CDF of h(x)
def inv_H(u):
    return math.sqrt(u*(math.pi**2)/2)

# Define f(x)
def f(x):
    y = np.zeros_like(x)
    mask1 = (x > math.pi/2) & (x <= math.pi)
    mask2 = (x >= 0) & (x <= math.pi/2)
    y[mask1] = 4/math.pi - 4*x[mask1]/math.pi**2
    y[mask2] = 4*x[mask2]/math.pi**2
    return y

# MAIN CODE
# Transform the sample to get a sample from f(x)
x_transformed = np.zeros(shape=(1000,))
iteration = 0
while iteration < 1000:
    temp = 0
    u = random.uniform(0, 1)
    if u <= 1/2:
        temp = math.sqrt(u/2)*math.pi
    else:
# Set 1-u and subtract the whole thing from pi
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

**METHOD OF REJECTION**

from scipy.optimize import minimize_scalar

res = minimize_scalar(lambda x: -(h(x)/g(x)), bounds=(0, math.pi/2), method='bounded')
supremum = -res.fun

# MAIN CODE
iteration = 0
samples = np.zeros(shape=(10000, ))

while iteration < 10000:
# Step 1: Generate a Y from g
    x = random.uniform(0, 1)

# Step 2: Generate u from U(0, 1)
    u = random.uniform(0, 1)
# Step 3: Check the interval and reject if necessary
    if u <= 1/2:
# Set y after we have decided the interval
        y = np.arccos(1 - x)
        if u <= (4*y)/(math.sin(y)*supremum*math.pi**2):
            samples[iteration] = y
            iteration += 1
        else:
            continue
    else:
# Set y equal to pi minus (1-x) which is x
        y = math.pi - np.arccos(x)
        if u <= (4*y)/(math.sin(y)*supremum*math.pi**2):
            samples[iteration] = y 
            iteration += 1
        else:
            continue

