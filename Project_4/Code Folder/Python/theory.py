import numpy as np
import math
from scipy.integrate import quad
from scipy.stats import ttest_1samp


# φ(x) = I[X>1]
def h(x, sigma):
    y = np.where(sigma*x >= 1, 1, 0)
    return y


# the initial integral
def to_b_integrated(x, sigma):
    return 1 * np.e**((-(x/sigma)**2)/2)/(sigma * np.sqrt(2*math.pi))


# used for the calculations of the variance of theta bar 1
def to_b_integrated_theta_1(x, theta, sigma):
    return abs(h(x, sigma)-theta)**2 * np.e**((-(x/sigma)**2)/2)/(sigma * np.sqrt(2*math.pi))


# used for the calculations of the variance of theta bar 2
def to_b_integrated_theta_2(x, theta, sigma):
    return abs((np.e**(-x**2*(1/((2*sigma**2) - 1/2)))*h(x, sigma))/sigma - theta)**2 * np.e**((-(x/sigma)**2)/2)/(sigma*np.sqrt(2*math.pi))


def monte_carlo(num_samples, sigma):

    # get a sample from the Normal Distribution
    samples = np.random.normal(loc=0, scale=1, size=num_samples)

    # calculate the theta bars
    h_i = h(samples, sigma)
    h_i_2 = h(samples, 1)
    theta_bar_1 = (np.sum(h_i)) / num_samples
    theta_bar_2 = (np.sum(np.e**(-samples**2 * (1/(2*(sigma**2)) - 1/2))*h_i_2)) / (num_samples * sigma)

    # calculate the exact value of the integral
    to_b_integrated_vec = np.vectorize(lambda x: to_b_integrated(x, sigma))
    theta, _ = quad(to_b_integrated_vec, 1, math.inf)

    # calculate the variance of theta bar 1
    to_b_integrated_vec_1 = np.vectorize(lambda x: to_b_integrated_theta_1(x, theta, 1))
    integral_1, _ = quad(to_b_integrated_vec_1, 1, math.inf)
    variance_1 = integral_1/num_samples

    # calculate the variance of theta bar 2
    to_b_integrated_vec_2 = np.vectorize(lambda x: to_b_integrated_theta_2(x, theta, 1))
    integral_2, _ = quad(to_b_integrated_vec_2, 1, math.inf)
    variance_2 = integral_2/num_samples

    print("The theta we simulated with theta 1: ", theta_bar_1)
    print("The theta we simulated with theta 2: ", theta_bar_2)
    print("The original theta: ", theta)
    print("The variance of theta 1: ", variance_1)
    print("The variance of theta 2: ", variance_2)
    # perform t-test
    t_stat, p_value = ttest_1samp(h_i, theta)
    print("The t-statistic: ", t_stat)
    print("The p-value: ", p_value)


# Choose number of samples and a sigma (better choose a sigma less and close to 1)
monte_carlo(10000, 0.5)
