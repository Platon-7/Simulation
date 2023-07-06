import numpy as np
import math
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')


# Note: There are 2 methods implemented, the first is with Rayleigh and the second with Gamma Distribution

# φ(x) for Rayleigh
def h1(x):
    return np.sin(np.pi*x) * x


# φ(x) for Gamma
def h2(x):
    return np.sin(np.pi * x) * 2 * np.e**((-x**2 + 2*x)/2)


# the initial integral
def to_b_integrated(x):
    return x**2 * np.e**((-x**2)/2) * math.sin(math.pi*x)


# used for the calculations of the variance of theta bar (for Rayleigh)
def to_b_integrated_2(x, theta):
    return (abs(np.sin(np.pi*x) * x - theta)**2) * (np.sin(np.pi * x) * x)


# used for the calculations of the variance of theta bar (for Gamma)
def to_b_integrated_3(x, theta):
    return (abs(np.sin(np.pi * x) * 2 * np.e**((-x**2 + 2*x)/2) - theta)**2) * ((np.e**(-x) * x**2)/2)


def monte_carlo_rayleigh(num_samples):

    # get a sample from the Rayleigh distribution with sigma 1
    samples = np.random.rayleigh(scale=1, size=num_samples)

    # calculate theta bar
    h_i = h1(samples)
    theta_bar = np.sum(h_i) / num_samples

    # calculate the exact value of the integral
    theta, _ = quad(to_b_integrated, 0, math.inf)

    # calculate the variance of theta bar
    to_b_integrated_2_vec = np.vectorize(lambda x: to_b_integrated_2(x, theta))
    integral_2, _ = quad(to_b_integrated_2_vec, 0, math.inf)
    variance = integral_2/num_samples

    # calculate the sample standard deviation and the confidence interval
    std_dev = np.std(h_i)
    conf_int = (theta_bar - 1.96 * std_dev / np.sqrt(num_samples), theta_bar + 1.96 * std_dev / np.sqrt(num_samples))

    print("The results of Rayleigh Distribution as a PDF function: ")
    print("The theta we simulated: ", theta_bar)
    print("The original theta: ", theta)
    print("The standard deviation: ", std_dev)
    print("The variance of theta bar: ", variance)
    print("The 95% confidence interval: ", conf_int)


def monte_carlo_gamma(num_samples):

    # get a sample from the Gamma distribution with k = 3 and θ = 1
    samples = np.random.gamma(3, 1, size=num_samples)

    # calculate theta bar
    h_i = h2(samples)
    theta_bar = np.sum(h_i) / num_samples

    # calculate the exact value of the integral
    theta, _ = quad(to_b_integrated, 0, math.inf)

    # calculate the variance of theta bar
    to_b_integrated_3_vec = np.vectorize(lambda x: to_b_integrated_3(x, theta))
    integral_2, _ = quad(to_b_integrated_3_vec, 0, math.inf)
    variance = integral_2/num_samples

    # calculate the sample standard deviation and the confidence interval
    std_dev = np.std(h_i)
    conf_int = (theta_bar - 1.96 * std_dev / np.sqrt(num_samples), theta_bar + 1.96 * std_dev / np.sqrt(num_samples))

    print("-------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------")
    print("The results of Gamma Distribution as a PDF function: ")
    print("The theta we simulated: ", theta_bar)
    print("The original theta: ", theta)
    print("The standard deviation: ", std_dev)
    print("The variance of theta bar: ", variance)
    print("The 95% confidence interval: ", conf_int)


monte_carlo_rayleigh(10000)
monte_carlo_gamma(10000)
