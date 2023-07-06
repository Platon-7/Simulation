import numpy as np
import math
from scipy.integrate import quad


# φ(x)
def h(x, normalization):
    return 2 * normalization * np.sin(2*np.pi*x) / x


# f(x) but without normalization
def un_norm(x):
    return (x/2) * np.e**(-x**2/4)


# the initial integral
def to_b_integrated(x):
    return np.e**((-x**2)/4) * math.sin(2*math.pi*x)


# used for the calculations of the variance of theta bar
def to_b_integrated_2(x, theta, normalization):
    return (abs((2 * normalization * np.sin(2*np.pi*x) / x) - theta))* *2 * (np.e* *((-x**2)/4) * (x/2) * normalization)


def monte_carlo(num_samples):

# normalize the Rayleigh distribution from (0, +oo) to (1, +oo)
    normalization, _ = quad(un_norm, 1, math.inf)
# get a sample from the Rayleigh distribution with sigma sqrt(2)
    samples = np.random.rayleigh(scale=np.sqrt(2), size=num_samples)

# calculate theta bar
    h_i = h(samples, normalization)
    theta_bar = np.sum(h_i) / num_samples

# calculate the exact value of the integral
    theta, _ = quad(to_b_integrated, 1, math.inf)

# calculate the variance of theta bar
    to_b_integrated_2_vec = np.vectorize(lambda x: to_b_integrated_2(x, theta, normalization))
    integral_2, _ = quad(to_b_integrated_2_vec, 1, math.inf)
    variance = integral_2/num_samples

# calculate the sample standard deviation and the confidence interval
    std_dev = np.std(h_i)
    conf_int = (theta_bar - 1.96 * std_dev / np.sqrt(num_samples), theta_bar + 1.96 * std_dev / np.sqrt(num_samples))

    print("The theta we simulated: ", theta_bar)
    print("The original theta: ", theta)
    print("The standard deviation: ", std_dev)
    print("The variance of theta bar: ", variance)
    print("The 95% confidence interval: ", conf_int)


monte_carlo(10000)
