import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skewnorm

sns.set()

root_dir = os.path.abspath(os.curdir)
root_dir = '/'.join(root_dir.split('/')[:-1])  # nav up 1 directory from '_code' folder

# parameter settings
np.random.seed(1)
mu = 0  # mean
sigma = 1  # standard deviation
n = 10000  # number of samples

# create distribution values
normal_values = np.random.normal(mu, sigma, n)

# plot histogram
_, bins, __ = plt.hist(normal_values, 20, density=True, alpha=0.8)
plt.title(f'Histogram of {n} Normal Distribution')
plt.ylabel('Probability Density')
plt.xlabel('Standard Deviation')

# plot probability density function on top of histogram
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp(-(bins - mu) ** 2 / (2 * sigma ** 2)),
         linewidth=2, color='r')
plt.savefig(root_dir + '/images/normal_distribution.png')
plt.show()


# create right skewed distribution
skew_factor = 4
rv = skewnorm(skew_factor)
x = np.linspace(skewnorm.ppf(0.000000001, skew_factor), skewnorm.ppf(0.9999999999, skew_factor), 100)
x -= 10
plt.plot(x, rv.pdf(x), 'k-', lw=2, color='r')
right_skewed = skewnorm.rvs(skew_factor, size=10000)
right_skewed -= 10
plt.hist(right_skewed, 20, density=True, alpha=0.8)
plt.title(f'Histogram of {n} Right Skewed Normal Distribution')
plt.ylabel('Probability Density')
plt.xlabel('Standard Deviation')
# plt.savefig(...)
plt.show()


# create left skewed distribution
skew_factor = -4
rv = skewnorm(skew_factor)
x = np.linspace(skewnorm.ppf(0.000000001, skew_factor), skewnorm.ppf(0.9999999999, skew_factor), 100)
x += 10
plt.plot(x, rv.pdf(x), 'k-', lw=2, color='r')
left_skewed = skewnorm.rvs(skew_factor, size=10000)
left_skewed += 10
plt.hist(left_skewed, 20, density=True, alpha=0.8)
plt.title(f'Histogram of {n} Left Skewed Normal Distribution')
plt.ylabel('Probability Density')
plt.xlabel('Standard Deviation')
# plt.savefig(...)
plt.show()
