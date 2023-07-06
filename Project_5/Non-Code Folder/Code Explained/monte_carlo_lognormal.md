import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


# Convert Normal Distribution N(0,1) samples to Log-Normal LN(mu,sigma)
samples_normal = np.random.normal(0, 1, size=1000)
samples_lognormal = np.exp(samples_normal)

log_normal_mean = np.mean(samples_lognormal)
log_normal_var = np.var(samples_lognormal)

plt.hist(samples_lognormal, bins=100)
plt.show


# Generate the corresponding mu, sigma for LN if you started with N(0,1)
mu = np.log(log_normal_mean**2 / np.sqrt(log_normal_var + log_normal_mean**2))
sigma = np.sqrt(np.log(log_normal_var / log_normal_mean**2 + 1))


# initial integral
def to_b_integrated(n, x):
    return x**n * np.e**(-x) * (1/(sigma*x*np.sqrt(2*np.pi)))*(np.e**((-(np.log(x)-mu)**2)/(2*sigma**2)))


# used for variance calculation
def to_b_integrated2(n, theta, x):
    return ((x**n * np.e**(-x) - theta)**2) * (1/(sigma*x*np.sqrt(2*np.pi)))*(np.e**((-(np.log(x)-mu)**2)/(2*sigma**2)))


# φ(x)
def h(x, n):
    return x**n * np.e**(-x)


def monte_carlo(samples, n):

# calculate theta bar
    h_i = h(samples, n)
    theta_bar = np.sum(h_i) / len(samples)

# calculate the exact value of the integral
    to_b_integrated_vec = np.vectorize(lambda x: to_b_integrated(n, x))
    theta, _ = quad(to_b_integrated_vec, 0, math.inf)

# calculate the variance of theta bar
    to_b_integrated_2_vec = np.vectorize(lambda x: to_b_integrated2(n, theta, x))
    integral, _ = quad(to_b_integrated_2_vec, 0, math.inf)
    variance = integral / len(samples)


# calculate the variance using control variates
    sum = 0
    my_list = np.zeros(10,) # create a list to place the b values
    counter = 0
    for i in samples[:10]:
        res_1 = minimize_scalar(lambda b: (h(i, n) - b*(((i**n * np.e**(-i))/math.factorial(n+1))**2 - i))**2, method='Brent')
        infimum_1 = res_1.fun # the b that minimizes the function
        my_list[counter] = infimum_1
        counter += 1

    b = np.mean(my_list) # get the mean of the 10 b values that minimized the function
    print("b is: ", b)
    for i in samples[:10]:
        sum += ((h(i, n) - b*(((i**n * np.e**(-i))/math.factorial(n+1)) - i))**2)/10
    control_variance = sum/len(samples)


    # calculate the sample standard deviation and the confidence interval
    std_dev = np.std(h_i)
    conf_int = (theta_bar - 1.96 * std_dev / np.sqrt(len(samples)), theta_bar + 1.96 * std_dev / np.sqrt(len(samples)))

    print("The theta we simulated: ", theta_bar)
    print("The original theta: ", theta)
    print("The variance of theta bar: ", variance)
    print("The control variates variance: ", control_variance)


monte_carlo(samples_lognormal, 6)








