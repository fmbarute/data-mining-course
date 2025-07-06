import numpy as np
import matplotlib.pyplot as plt

# (c)-(g) Probability that jth observation is in bootstrap sample
n_values = np.arange(1, 100001)
prob_in_bootstrap = 1 - (1 - 1/n_values)**n_values

plt.figure(figsize=(10, 6))
plt.plot(n_values, prob_in_bootstrap)
plt.xlabel('n')
plt.ylabel('Probability jth observation is in bootstrap sample')
plt.title('Bootstrap Inclusion Probability')
plt.grid(True)
plt.show()

# For specific n values:
for n in [5, 100, 10000]:
    prob = 1 - (1 - 1/n)**n
    print(f"n={n}: Probability = {prob:.4f}")

# (h) Numerical investigation
rng = np.random.default_rng(10)
store = np.empty(10000)
for i in range(10000):
    store[i] = np.sum(rng.choice(100, replace=True, size=100) == 4)
print(f"Empirical probability: {np.mean(store > 0):.4f}")  # Fixed: Check if count > 0 before taking mean