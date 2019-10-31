import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import gamma, skewnorm, skew

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
sns.distplot(normal_values, bins=20)
plt.title('Normal Distribution')
plt.ylabel('Probability (PDF)')
plt.savefig(root_dir + '/images/normal_distribution.png')
plt.show()

dist_mean = np.round(normal_values.mean(), 4)
dist_std = np.round(normal_values.std(), 4)
dist_skew = np.round(skew(normal_values), 4)
print(f'Normal Skewed mean: {dist_mean}')
print(f'Normal Skewed  std: {dist_std}')
print(f'Normal Skewed skew: {dist_skew}\n')

# plot left skewed box plot
sns.boxplot(normal_values)
plt.title('Normal Distribution Box Plot')
plt.savefig(root_dir + '/images/left_skewed_boxplot.png')
plt.show()




# create right skewed distribution
skew_factor = 4
right_skewed_values = gamma.rvs(a=skew_factor, size=10000)
sns.distplot(right_skewed_values, kde=True, bins=20)
plt.title(f'Right Skewed Distribution')
plt.ylabel('Probability (PDF)')
plt.savefig(root_dir + '/images/right_skewed_distribution.png')
plt.show()

dist_mean = np.round(right_skewed_values.mean(), 4)
dist_std = np.round(right_skewed_values.std(), 4)
dist_skew = np.round(skew(right_skewed_values), 4)
print(f'Right Skewed mean: {dist_mean}')
print(f'Right Skewed  std: {dist_std}')
print(f'Right Skewed skew: {dist_skew}\n')


# create left skewed distribution
skew_factor = -6
left_skewed_values = skewnorm.rvs(skew_factor, size=10000)
left_skewed_values += (-left_skewed_values.min() + 1)
sns.distplot(left_skewed_values, bins=20)
plt.title('Left Skewed Distribution')
plt.ylabel('Probability (PDF)')
plt.savefig(root_dir + '/images/left_skewed_distribution.png')
plt.show()

dist_mean = np.round(left_skewed_values.mean(), 4)
dist_std = np.round(left_skewed_values.std(), 4)
dist_skew = np.round(skew(left_skewed_values), 4)
print(f'Left Skewed mean: {dist_mean}')
print(f'Left Skewed  std: {dist_std}')
print(f'Left Skewed skew: {dist_skew}\n')


# plot left skewed box plot
sns.boxplot(left_skewed_values)
plt.title('Left Skewed Distribution Box Plot')
plt.savefig(root_dir + '/images/left_skewed_boxplot.png')
plt.show()

