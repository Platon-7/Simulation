import numpy as np
import math
from scipy.integrate import quad
import random


# φ(x)
def h(x):
    return np.sqrt(1-x**2)


# the initial integral
def to_b_integrated(x):
    return np.sqrt(1 - x**2)

# used for the calculations of the variance of theta bar
def to_b_integrated_2(x, theta):
    return (abs(np.sqrt(1-x* *2) - theta)**2) * 1


def monte_carlo(num_samples):

# get a sample from the Uniform distribution in (0,1)
    samples = np.zeros((2*num_samples,))
    for i in range(num_samples):
        x = random.uniform(0, 1)
        samples[i] = x

# get an equal amount of samples for (-1,0) by getting the opposite values of the samples you just generated
    counter1 = 0
    counter2 = num_samples - 1
    while counter2 < num_samples*2:
        samples[counter2] = -samples[counter1]
        counter1 += 1
        counter2 += 1

# calculate theta bar
    h_i = h(samples)
    theta_bar =  np.sum(h_i) / num_samples

# calculate the exact value of the integral
    theta, _ = quad(to_b_integrated, -1, 1)

# calculate the variance of theta bar
# Note: Here we calculate the variance of the 1 half and multiply it by 2 since it is symmetrical around 0
    to_b_integrated_2_vec = np.vectorize(lambda x: to_b_integrated_2(x, theta/2))
    integral_2, _ = quad(to_b_integrated_2_vec, 0, 1)
    variance = 2 * (integral_2 / num_samples)

# calculate the sample standard deviation and the confidence interval
    std_dev = np.std(h_i)
    conf_int = (theta_bar - 1.96 * std_dev / np.sqrt(num_samples), theta_bar + 1.96 * std_dev / np.sqrt(num_samples))

    print("The theta we simulated: ", theta_bar)
    print("The original theta: ", theta)
    print("The standard deviation: ", std_dev)
    print("The variance of theta bar: ", variance)
    print("The 95% confidence interval: ", conf_int)


monte_carlo(10000)
