import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class QDABayesClassifier:
    """
    Implements a Quadratic Discriminant Analysis (QDA) Bayes classifier for one feature
    where each class has its own mean and variance.
    """

    def __init__(self, K, means, variances, priors):
        """
        Initialize the QDA Bayes classifier.

        Parameters:
        - K: number of classes
        - means: list of means for each class (length K)
        - variances: list of variances for each class (length K)
        - priors: list of prior probabilities for each class (length K)
        """
        self.K = K
        self.means = np.array(means)
        self.variances = np.array(variances)
        self.priors = np.array(priors)

    def discriminant_function(self, x, k):
        """
        Compute the discriminant function δ_k(x) for class k at point x.

        The discriminant function is derived from the log of the posterior probability:
        δ_k(x) = log(f_k(x)) + log(π_k)
        where f_k(x) is the normal density for class k.
        """
        # Compute the log of the normal density
        log_density = -0.5 * np.log(2 * np.pi * self.variances[k]) \
                      - 0.5 * (x - self.means[k]) ** 2 / self.variances[k]

        # Add the log prior
        return log_density + np.log(self.priors[k])

    def classify(self, x):
        """
        Classify observation x using the Bayes classifier.

        Returns:
        - The class with the highest posterior probability
        - The posterior probabilities for all classes
        """
        # Compute discriminant functions for all classes
        discriminants = [self.discriminant_function(x, k) for k in range(self.K)]

        # The class with the maximum discriminant value
        predicted_class = np.argmax(discriminants)

        # Convert discriminants to posterior probabilities (using softmax)
        max_d = np.max(discriminants)
        exp_d = np.exp(discriminants - max_d)  # for numerical stability
        posteriors = exp_d / np.sum(exp_d)

        return predicted_class, posteriors

    def plot_decision_boundary(self, x_range=(-10, 10), resolution=1000):
        """
        Plot the decision boundary and class densities.
        """
        x = np.linspace(x_range[0], x_range[1], resolution)

        # Compute discriminant functions for all x values
        discriminants = np.array([[self.discriminant_function(xi, k) for xi in x]
                                  for k in range(self.K)])

        # Find the decision boundaries (where two classes have equal discriminant values)
        decision_points = []
        for i in range(self.K):
            for j in range(i + 1, self.K):
                # Find where discriminant i equals discriminant j
                diff = discriminants[i] - discriminants[j]
                crossings = np.where(diff[:-1] * diff[1:] <= 0)[0]

                for idx in crossings:
                    # Linear interpolation to find exact crossing point
                    x1, x2 = x[idx], x[idx + 1]
                    d1, d2 = diff[idx], diff[idx + 1]
                    if d1 == d2:
                        x_cross = x1
                    else:
                        x_cross = x1 - d1 * (x2 - x1) / (d2 - d1)
                    decision_points.append(x_cross)

        # Plot the class densities
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for k in range(self.K):
            density = norm.pdf(x, self.means[k], np.sqrt(self.variances[k])) * self.priors[k]
            plt.plot(x, density, color=colors[k % len(colors)],
                     label=f'Class {k} (μ={self.means[k]}, σ²={self.variances[k]:.2f})')

        # Plot the decision boundaries
        for boundary in decision_points:
            plt.axvline(x=boundary, color='black', linestyle='--', alpha=0.7)

        plt.title('QDA Class Densities and Decision Boundaries')
        plt.xlabel('Feature value (x)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return decision_points


# Proof that the Bayes classifier is quadratic when variances are unequal

def prove_quadratic_decision():
    """
    Mathematical proof that the decision boundary is quadratic when variances differ.
    """
    print("Proof that the Bayes classifier is quadratic when σ² differs between classes:")
    print("=" * 70)
    print("\nConsider two classes (k=1,2) with different variances (σ₁² ≠ σ₂²).")
    print("The discriminant functions are:")
    print("δ₁(x) = -½log(2πσ₁²) - (x-μ₁)²/(2σ₁²) + log(π₁)")
    print("δ₂(x) = -½log(2πσ₂²) - (x-μ₂)²/(2σ₂²) + log(π₂)")

    print("\nThe decision boundary occurs where δ₁(x) = δ₂(x):")
    print("-½log(2πσ₁²) - (x-μ₁)²/(2σ₁²) + log(π₁) = -½log(2πσ₂²) - (x-μ₂)²/(2σ₂²) + log(π₂)")

    print("\nSimplify and rearrange terms:")
    print("(x-μ₂)²/(2σ₂²) - (x-μ₁)²/(2σ₁²) = ½log(σ₁²/σ₂²) + log(π₁/π₂)")

    print("\nMultiply both sides by 2σ₁²σ₂²:")
    print("σ₁²(x-μ₂)² - σ₂²(x-μ₁)² = σ₁²σ₂²[log(σ₁²/σ₂²) + 2log(π₁/π₂)]")

    print("\nExpand the squared terms:")
    print("σ₁²(x² - 2μ₂x + μ₂²) - σ₂²(x² - 2μ₁x + μ₁²) = C")
    print(f"Where C = σ₁²σ₂²[log(σ₁²/σ₂²) + 2log(π₁/π₂)]")

    print("\nCombine like terms:")
    print(f"(σ₁² - σ₂²)x² + (-2σ₁²μ₂ + 2σ₂²μ₁)x + (σ₁²μ₂² - σ₂²μ₁² - C) = 0")

    print("\nThis is a quadratic equation of the form ax² + bx + c = 0,")
    print("which proves the decision boundary is quadratic when σ₁² ≠ σ₂².")
    print("When σ₁² = σ₂², the x² terms cancel out, leaving a linear boundary.")


# Example usage
if __name__ == "__main__":
    # Example with 2 classes having different variances
    K = 2
    means = [1, 3]
    variances = [1, 4]  # Different variances
    priors = [0.5, 0.5]

    qda = QDABayesClassifier(K, means, variances, priors)

    # Plot the decision boundary
    print("Visualizing QDA with unequal variances:")
    decision_points = qda.plot_decision_boundary(x_range=(-5, 8))
    print(f"Decision boundaries at: {decision_points}")

    # Classify some example points
    test_points = [-2, 0, 1, 2, 4, 6]
    print("\nClassification examples:")
    for x in test_points:
        pred_class, posteriors = qda.classify(x)
        print(f"x = {x:.1f}: Class {pred_class} (Posteriors: {posteriors.round(3)})")

    # Mathematical proof
    prove_quadratic_decision()