import numpy as np
from ISLP import load_data

# Load Boston dataset
Boston = load_data('Boston')

# (a) Estimate mean of medv
mu_hat = Boston['medv'].mean()

# (b) Standard error of mean
se_formula = Boston['medv'].std() / np.sqrt(len(Boston))

# (c) Bootstrap standard error
np.random.seed(1)
boot_means = []
for _ in range(1000):
    sample = Boston['medv'].sample(n=len(Boston), replace=True)
    boot_means.append(sample.mean())
boot_se = np.std(boot_means)

# (d) Confidence intervals
ci_formula = [mu_hat - 2*se_formula, mu_hat + 2*se_formula]
ci_boot = [mu_hat - 2*boot_se, mu_hat + 2*boot_se]

# (e) Median estimate
mu_med = Boston['medv'].median()

# (f) Bootstrap SE for median
boot_medians = []
for _ in range(1000):
    sample = Boston['medv'].sample(n=len(Boston), replace=True)
    boot_medians.append(sample.median())
boot_med_se = np.std(boot_medians)

# (g) 10th percentile
mu_01 = np.percentile(Boston['medv'], 10)

# (h) Bootstrap SE for 10th percentile
boot_01 = []
for _ in range(1000):
    sample = Boston['medv'].sample(n=len(Boston), replace=True)
    boot_01.append(np.percentile(sample, 10))
boot_01_se = np.std(boot_01)

# Print results
print(f"(a) Mean estimate: {mu_hat:.4f}")
print(f"(b) Formula SE: {se_formula:.4f}")
print(f"(c) Bootstrap SE: {boot_se:.4f}")
print(f"(d) Formula CI: [{ci_formula[0]:.4f}, {ci_formula[1]:.4f}]")
print(f"    Bootstrap CI: [{ci_boot[0]:.4f}, {ci_boot[1]:.4f}]")
print(f"(e) Median estimate: {mu_med:.4f}")
print(f"(f) Bootstrap SE for median: {boot_med_se:.4f}")
print(f"(g) 10th percentile: {mu_01:.4f}")
print(f"(h) Bootstrap SE for 10th percentile: {boot_01_se:.4f}")