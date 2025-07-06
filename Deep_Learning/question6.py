import numpy as np
import matplotlib.pyplot as plt


def R(beta):
    return np.sin(beta) + beta / 10


def dR(beta):
    return np.cos(beta) + 1 / 10


def gradient_descent(start, lr=0.1, n_iter=20):
    beta = start
    history = [beta]
    for _ in range(n_iter):
        beta = beta - lr * dR(beta)
        history.append(beta)
    return history


# Plot function
betas = np.linspace(-6, 6, 100)
plt.plot(betas, R(betas), label='R(β)')

# Run GD from β=2.3
history1 = gradient_descent(2.3)
plt.scatter(history1, R(np.array(history1)), c='r', label='β=2.3 path')
print("From β=2.3, converges to:", history1[-1])

# Run GD from β=1.4
history2 = gradient_descent(1.4)
plt.scatter(history2, R(np.array(history2)), c='g', label='β=1.4 path')
print("From β=1.4, converges to:", history2[-1])

plt.xlabel('β')
plt.ylabel('R(β)')
plt.legend()
plt.grid()
plt.show()
