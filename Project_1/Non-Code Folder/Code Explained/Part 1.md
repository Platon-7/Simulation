These chunks contain code from the whole 1st exercise


**METHOD OF INVERSION**

# Definining the functions
def f(x):
    y = np.zeros_like(x)
    mask1 = (x > math.pi/2) & (x <= math.pi) # this is a check for the interval
    mask2 = (x >= 0) & (x <= math.pi/2) # and for the other interval
    y[mask1] = 4/math.pi - 4*x[mask1]/math.pi**2
    y[mask2] = 4*x[mask2]/math.pi**2
    return y
	
	
# MAIN CODE
iteration = 0
samples = np.zeros(shape=(1000, )) # create an np array with 1000 values
x = 0 # define x

while iteration < 1000:
    
    # Step 1: Generate a uniform value u in (0,1)
    u = random.uniform(0, 1)
    # Check in which interval it belongs
    if u > 1/2:
        x1 = math.pi*((2 + math.sqrt(2-2*u))/2) # the two x's that were computed in theory
        x2 = math.pi*((2 - math.sqrt(2-2*u))/2)
        # One of the x's should be rejected since F is '1-1' in [pi/2, pi]
        if x1 >= math.pi/2 and x1 <= math.pi:
            x = x1
        else:
            x = x2
        
    else:
        x = math.sqrt(u/2)*math.pi

    samples[iteration] = x # 
    iteration += 1
	
	
# Plot the histogram
plt.hist(samples, bins=30, density=True, alpha=0.8, label='Histogram')

# Estimate the PDF curve using KDE
kde = gaussian_kde(samples)
z = np.linspace(samples.min(), samples.max(), 1000) # get 1000 values equally distributed 
estimated_pdf = kde(z)
plt.plot(z, estimated_pdf, 'r-', label='PDF') # this plots the estimating PDF 

# Plot the true PDF curve
real_pdf = [f(xi) for xi in z]
plt.plot(z, real_pdf, 'y--', label='True PDF') # this plots the original PDF

# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('PDF')
plt.legend()

# Show the plot
plt.show()	



**METHOD OF REJECTION**

def f(x):
    y = np.zeros_like(x)
    mask1 = (x > math.pi/2) & (x <= math.pi)
    mask2 = (x >= 0) & (x <= math.pi/2)
    y[mask1] = 4/math.pi - 4*x[mask1]/math.pi**2
    y[mask2] = 4*x[mask2]/math.pi**2
    return y

def g(x):
    return math.sin(x)/2 # define the envelope function


# Calculate the supremum for each interval (even though they are equal in this case)
from scipy.optimize import minimize_scalar

res_1= minimize_scalar(lambda x: -(f(x)/g(x)), bounds=(0, math.pi/2), method='bounded')
supremum_1 = -res_1.fun

res_2 = minimize_scalar(lambda x: -(f(x)/g(x)), bounds=(math.pi/2, math.pi), method='bounded')
supremum_2 = -res_2.fun

iteration = 0
samples = np.zeros(shape=(10000, ))


# MAIN CODE 
while iteration < 10000:
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
# Note: The iteration variable increases only in success to avoid null values

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