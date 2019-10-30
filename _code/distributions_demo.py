import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# parameter settings
np.random.seed(1)
mu = 0  # mean
sigma = 1  # standard deviation
n = 10000  # number of samples

# create distribution values
normal_values = np.random.normal(mu, sigma, n)

# plot histogram
_, bins, __ = plt.hist(normal_values, 50, density=True)
plt.title(f'Histogram of {n} random Normal Distribution samples')
plt.ylabel('Probability Density')
plt.xlabel('Standard Deviation')

# plot probability density function on top of histogram
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp(-(bins - mu) ** 2 / (2 * sigma ** 2)),
         linewidth=2, color='r')
plt.show()
