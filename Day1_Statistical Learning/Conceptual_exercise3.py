import matplotlib.pyplot as plt
import numpy as np


def plot_bias_variance():
    x = np.linspace(0, 10, 100)
    plt.figure(figsize=(10, 6))

    # Example curves (simplified)
    plt.plot(x, np.exp(-x), label='Bias²')
    plt.plot(x, np.log(x + 1), label='Variance')
    plt.plot(x, np.exp(-x) + np.log(x + 1), label='Test Error')

    plt.legend()
    plt.xlabel('Model Flexibility')
    plt.ylabel('Error')
    plt.title('Bias-Variance Tradeoff')
    plt.savefig('bias_variance.png')


if __name__ == "__main__":
    plot_bias_variance()