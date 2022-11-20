import numpy as np
import matplotlib.pyplot as plt


def weibull(_lambda, k):
    def f(x):
        return k / _lambda * ((x / _lambda) ** (k-1)) * np.exp(-((x / _lambda) ** k))
    return f


x = np.arange(0, 1, 1e-3)
y = np.vectorize(weibull(0.25, 1.7))(x)
print(y)

plt.plot(x, y)
plt.show()
