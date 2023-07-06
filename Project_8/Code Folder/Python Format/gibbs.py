# Install packages

import subprocess

subprocess.call(['pip', 'install', 'statsmodels'])
subprocess.call(['pip', 'install', 'openpyxl'])
subprocess.call(['pip', 'install', 'arspy'])

### Import Files

import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from arspy.ars import adaptive_rejection_sampling


### Convert Excel --> CSV

# Read Excel file
excel_file = pd.read_excel('simulation_data.xlsx')

# Convert to CSV
csv_file = 'simulation_data.csv'
excel_file.to_csv(csv_file, index=False)

print(f"Excel file converted to CSV: {csv_file}")

### Preview the Data

df = pd.read_csv(csv_file)
print(df)

### Calculate the Beta_Hat's and the Covariance Matrix

# Perform one-hot encoding on relevant columns
one_hot_encoded_data = pd.get_dummies(df, columns=['Obesity', 'Hypertension', 'Alcohol Consumption per Day'])

# Select the desired columns for the result
result_columns = ['Obesity_Average', 'Obesity_High', 'Hypertension_Yes',
                  'Alcohol Consumption per Day_1-2 Drinks', 'Alcohol Consumption per Day_3-5 Drinks', 'Alcohol Consumption per Day_6+ Drinks']
result_data = one_hot_encoded_data[result_columns]

# Convert the result dataframe to a numpy array
matrix = result_data.to_numpy()

# Convert True and False to 1 and 0
matrix = matrix.astype(int)

# Add an intercept column of ones
intercept_column = np.ones((matrix.shape[0], 1), dtype=int)
matrix = np.hstack((intercept_column, matrix))

# Print the matrix with clear 1s and 0s
np.set_printoptions(edgeitems=3, formatter={'int': '{:3d}'.format})
print(matrix)

# observations
y = df['Observations']

poisson_model = sm.GLM(y, matrix, family=sm.families.Poisson())  # Fit the Poisson GLM
results = poisson_model.fit()

beta_hat = results.params
cov_matrix_hat = results.cov_params()

print(beta_hat)
print(cov_matrix_hat.shape)


### Gibbs Sampling

def conditional_sampler(sampling_index, current_x, mean, obs):
    cond_indices = [i for i in range(len(mean)) if i != sampling_index]
    new_x = np.copy(current_x)
    temp = np.zeros((matrix.shape[0], 1))
    for i in range(matrix.shape[0]):
        temp[i] = np.sum(current_x[cond_indices] @ matrix[i][cond_indices])

    formula = lambda x: sum(-np.exp(matrix[i][sampling_index]*x + temp[i]) + obs[i] * (matrix[i][sampling_index]*x + temp[i]) for i in range(matrix.shape[0]))

    domain = (float("-inf"), float("inf"))
    a = -4
    b = 4

    new_x[sampling_index] = adaptive_rejection_sampling(formula, a, b, domain, 1)[0]
    return new_x
    
def gibbs_sampler(initial_point, num_samples, mean, burn_in, thinning, obs):

    point = np.array(initial_point)
    samples = np.zeros((num_samples, 7))  # sampled points

    sample_counter = 0
    counter = 0
    while sample_counter < num_samples:
        for i in range(len(mean)):
            point = conditional_sampler(i, point, mean, obs)

        # Save the sample using burn_in and thinning
        if counter > burn_in and (counter-burn_in) % thinning == 0:
            samples[sample_counter] = point
            sample_counter += 1

        counter += 1

    return samples
    
mean = np.array(beta_hat).T
initial_point = [2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1]
num_samples = 500
burn_in = 500
thinning = 1
observations = np.array(y)

samples = gibbs_sampler(initial_point, num_samples, mean, burn_in, thinning, observations)
sample_mean = np.mean(samples, axis = 0)
sample_std = np.std(samples, axis = 0)

print("The simulated mean is: {}".format(np.around(sample_mean, decimals=2)))

print("The simulated standard deviation is: {}".format(np.around(sample_std, decimals=2)))

### The Histograms of Gibbs for each Beta_Hat

# Get the number of dimensions
num_dimensions = samples.shape[1]

# Create a figure with subplots for each dimension
fig, axes = plt.subplots(nrows=num_dimensions, ncols=1, figsize=(6, num_dimensions*4))

# Iterate over each dimension
for i in range(num_dimensions):
    # Get the samples for the current dimension
    dimension_samples = samples[:, i]

    # Plot the histogram
    axes[i].hist(dimension_samples, bins=30, color='skyblue', edgecolor='black')

    # Add a vertical line for the true value
    axes[i].axvline(x=beta_hat[i], color='red', linestyle='--', linewidth=2)

    # Set the plot title and labels
    axes[i].set_title(f'Beta_Hat - {i+1}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

### Gibbs Convergence Chains

# Generate an array of iteration numbers
iterations = np.arange(1, num_samples + 1)

# Plot the convergence of the chain
plt.plot(iterations, samples[:, 0], color='blue', label='Beta_hat 1')
plt.plot(iterations, samples[:, 1], color='red', label='Beta_hat 2')
plt.plot(iterations, samples[:, 2], color='green', label='Beta_hat 3')
plt.plot(iterations, samples[:, 3], color='yellow', label='Beta_hat 4')
plt.plot(iterations, samples[:, 4], color='purple', label='Beta_hat 5')
plt.plot(iterations, samples[:, 5], color='pink', label='Beta_hat 6')
plt.plot(iterations, samples[:, 6], color='orange', label='Beta_hat 7')
# Add more lines for additional dimensions

# Set the plot title and labels
plt.title('Gibbs Chain Convergence')
plt.xlabel('Iteration')
plt.ylabel('Value')

# Add a legend
plt.legend()

# Show the plot
plt.show()


### The Autocorrelation Chains

from statsmodels.graphics.tsaplots import plot_acf

# Plot the autocorrelation for each dimension
fig, axes = plt.subplots(nrows=num_dimensions, ncols=1, figsize=(6, num_dimensions*4))

# Iterate over each dimension
for i in range(num_dimensions):
    # Get the samples for the current dimension
    dimension_samples = samples[:, i]

    # Plot the autocorrelation
    plot_acf(dimension_samples, ax=axes[i])

    # Set the plot title and labels
    axes[i].set_title(f'Autocorrelation - Beta_hat {i+1}')
    axes[i].set_xlabel('Lag')
    axes[i].set_ylabel('Autocorrelation')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

### Effective Sample Size Calculation

#import tensorflow_probability as tfp
#import tensorflow as tf

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#ess = tfp.mcmc.effective_sample_size(samples)[0]

#print(int(ess.numpy()))