import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline
plt.style.use('seaborn-poster')
from scipy.integrate import quad
from scipy.optimize import root_scalar


a = -1
b = 2


f = lambda x: norm.pdf(x)


result, error = quad(f, a, b)

print("The integral of the normal distribution function from", a, "to", b, "is:", result)

x = np.linspace(-1, 2, 100)
y =  [f(xi) for xi in x]


cs = CubicSpline(x, y)


a = -1
b = 2


integrand = lambda x: cs(x)


result, error = quad(integrand, a, b)


envelope_function = lambda x: cs(x) / result


a_norm = -1
b_norm = 2


g = lambda x: envelope_function(x)


result_norm, error_norm = quad(g, a_norm, b_norm)


print("The integral of the normalized envelope function from", a_norm, "to", b_norm, "is:", result_norm)

h = lambda x: f(x)/g(x)
res = minimize_scalar(lambda x: -(h(x)), bounds=(-1, 2), method='bounded')


supremum = 1/-res.fun
print(supremum)


F = lambda x: quad(g, -1, x)[0]


def G(y):

    for i in range(len(x)-1):
        a, b = x[i], x[i+1]
        if (F(a) - y) * (F(b) - y) <= 0:
            break
    
    return root_scalar(lambda x: F(x) - y, bracket=[a, b], method='bisect').root


F_inv = lambda y: root_scalar(lambda x: G(x) - y, bracket=[-1, 2], method='bisect').root

samples = np.zeros(shape=(10000, ))
print("Don't expect from this to run, it has an error that I didn't figure out, for some reason the estimation of F_inv which uses the bisection method doesn't work")
iteration = 0
while iteration < 10000:
    u1 = random.uniform(0, 1)
    u2 = random.uniform(0, 1)
    y1 = F_inv(u1)

    if u2 <= h(y1)/supremum:
            samples[iteration] = y1
            iteration += 1
    else:
        continue
		
z = np.linspace(-1, 2, 1000)
f_y = [f(xi) for xi in z]
g_y = [g(xi) for xi in z]

plt.plot(z, f_y, label='f(x)', color = 'b')

plt.plot(z, g_y, label='g(x)', color = 'g')

plt.hist(samples, bins=30, density=True, color = 'r', alpha=0.8, label='Rejection')

plt.title('PDFs')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1,2)
plt.ylim(0,3)
plt.legend()
plt.show()