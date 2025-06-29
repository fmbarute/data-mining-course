import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample


def load_auto_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                    'acceleration', 'model_year', 'origin', 'car_name']
    auto_data = pd.read_csv(url, delim_whitespace=True, names=column_names, na_values='?')
    auto_data = auto_data.dropna().drop('car_name', axis=1)
    return auto_data


def bootstrap_polynomial_evaluation(X, y, max_degree=5, n_bootstrap=1000, test_size=0.3, random_state=42):
    """
    Evaluate polynomial models using bootstrap sampling

    Parameters:
    - X, y: features and target
    - max_degree: maximum polynomial degree to test
    - n_bootstrap: number of bootstrap samples
    - test_size: proportion of data held out as test set
    - random_state: for reproducibility
    """

    # First split: hold out a test set that we never touch during model selection
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Store results for each degree
    all_results = []

    print(f"Running bootstrap evaluation with {n_bootstrap} samples...")
    print("=" * 60)

    for degree in range(1, max_degree + 1):
        print(f"Evaluating polynomial degree {degree}...")

        # Store bootstrap results for this degree
        bootstrap_train_scores = []
        bootstrap_oob_scores = []

        for i in range(n_bootstrap):
            # Create bootstrap sample from training data
            X_boot, y_boot = resample(X_temp, y_temp, random_state=i)

            # Find out-of-bag (OOB) samples - samples not in bootstrap
            boot_indices = set(resample(range(len(X_temp)), random_state=i))
            oob_indices = [idx for idx in range(len(X_temp)) if idx not in boot_indices]

            if len(oob_indices) == 0:  # Skip if no OOB samples
                continue

            X_oob = X_temp[oob_indices]
            y_oob = y_temp[oob_indices]

            # Train model on bootstrap sample
            model = make_pipeline(
                PolynomialFeatures(degree=degree),
                LinearRegression()
            )
            model.fit(X_boot, y_boot)

            # Evaluate on bootstrap sample (training error)
            y_boot_pred = model.predict(X_boot)
            train_mse = mean_squared_error(y_boot, y_boot_pred)
            bootstrap_train_scores.append(train_mse)

            # Evaluate on out-of-bag samples (validation error)
            y_oob_pred = model.predict(X_oob)
            oob_mse = mean_squared_error(y_oob, y_oob_pred)
            bootstrap_oob_scores.append(oob_mse)

        # Calculate statistics for this degree
        mean_train_mse = np.mean(bootstrap_train_scores)
        std_train_mse = np.std(bootstrap_train_scores)
        mean_oob_mse = np.mean(bootstrap_oob_scores)
        std_oob_mse = np.std(bootstrap_oob_scores)

        # Train final model on all training data for this degree
        final_model = make_pipeline(
            PolynomialFeatures(degree=degree),
            LinearRegression()
        )
        final_model.fit(X_temp, y_temp)

        # Evaluate on held-out test set
        y_test_pred = final_model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)

        result = {
            'Degree': degree,
            'Bootstrap_Train_MSE_Mean': mean_train_mse,
            'Bootstrap_Train_MSE_Std': std_train_mse,
            'Bootstrap_OOB_MSE_Mean': mean_oob_mse,
            'Bootstrap_OOB_MSE_Std': std_oob_mse,
            'Test_MSE': test_mse,
            'Model': final_model,
            'Bootstrap_Train_Scores': bootstrap_train_scores,
            'Bootstrap_OOB_Scores': bootstrap_oob_scores
        }

        all_results.append(result)

        print(f"  Bootstrap Train MSE: {mean_train_mse:.2f} ± {std_train_mse:.2f}")
        print(f"  Bootstrap OOB MSE: {mean_oob_mse:.2f} ± {std_oob_mse:.2f}")
        print(f"  Final Test MSE: {test_mse:.2f}")
        print("-" * 40)

    return pd.DataFrame(all_results), X_test, y_test


def plot_bootstrap_results(results_df):
    """Plot bootstrap evaluation results with confidence intervals"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    degrees = results_df['Degree']

    # Plot 1: MSE vs Degree with error bars
    ax1.errorbar(degrees, results_df['Bootstrap_Train_MSE_Mean'],
                 yerr=results_df['Bootstrap_Train_MSE_Std'],
                 marker='o', label='Bootstrap Training MSE', capsize=5)
    ax1.errorbar(degrees, results_df['Bootstrap_OOB_MSE_Mean'],
                 yerr=results_df['Bootstrap_OOB_MSE_Std'],
                 marker='s', label='Bootstrap OOB MSE', capsize=5)
    ax1.plot(degrees, results_df['Test_MSE'], 'r^-', label='Final Test MSE', markersize=8)

    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Bootstrap Model Evaluation with Confidence Intervals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(degrees)

    # Plot 2: Box plots of bootstrap scores for a few degrees
    selected_degrees = [1, 2, 3, 5] if len(degrees) >= 4 else degrees.tolist()
    oob_data = []
    labels = []

    for degree in selected_degrees:
        if degree in degrees.values:
            idx = degrees[degrees == degree].index[0]
            oob_scores = results_df.iloc[idx]['Bootstrap_OOB_Scores']
            oob_data.append(oob_scores)
            labels.append(f'Degree {degree}')

    if oob_data:
        ax2.boxplot(oob_data, labels=labels)
        ax2.set_ylabel('OOB MSE')
        ax2.set_title('Distribution of Bootstrap OOB Scores')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_best_model_with_uncertainty(X, y, best_model, degree, X_test, y_test, n_bootstrap=100):
    """Plot the best model with prediction uncertainty bands"""

    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

    # Generate bootstrap predictions for uncertainty estimation
    bootstrap_predictions = []

    for i in range(n_bootstrap):
        # Create bootstrap sample
        X_boot, y_boot = resample(X, y, random_state=i)

        # Train model on bootstrap sample
        boot_model = make_pipeline(
            PolynomialFeatures(degree=degree),
            LinearRegression()
        )
        boot_model.fit(X_boot, y_boot)

        # Generate predictions
        y_pred_boot = boot_model.predict(X_plot)
        bootstrap_predictions.append(y_pred_boot)

    # Calculate prediction statistics
    bootstrap_predictions = np.array(bootstrap_predictions)
    mean_pred = np.mean(bootstrap_predictions, axis=0)
    std_pred = np.std(bootstrap_predictions, axis=0)

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot data points
    plt.scatter(X, y, alpha=0.5, color='lightblue', label='Training Data')
    plt.scatter(X_test, y_test, alpha=0.7, color='orange', label='Test Data')

    # Plot main prediction line
    y_main_pred = best_model.predict(X_plot)
    plt.plot(X_plot, y_main_pred, 'r-', linewidth=2, label=f'Best Model (Degree {degree})')

    # Plot uncertainty bands
    plt.fill_between(X_plot.ravel(),
                     mean_pred - 2 * std_pred,
                     mean_pred + 2 * std_pred,
                     alpha=0.2, color='red', label='95% Prediction Interval')

    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.title(f'Bootstrap Model with Uncertainty (Degree {degree})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Main execution
if __name__ == "__main__":
    # Load and prepare data
    auto_data = load_auto_data()
    X = auto_data['horsepower'].values.reshape(-1, 1)
    y = auto_data['mpg'].values

    # Run bootstrap evaluation
    results_df, X_test, y_test = bootstrap_polynomial_evaluation(
        X, y, max_degree=5, n_bootstrap=200, random_state=42
    )

    # Find best model based on OOB MSE
    best_idx = results_df['Bootstrap_OOB_MSE_Mean'].idxmin()
    best_model_info = results_df.iloc[best_idx]

    print("\n" + "=" * 60)
    print("BEST MODEL SUMMARY:")
    print("=" * 60)
    print(f"Best Polynomial Degree: {best_model_info['Degree']}")
    print(
        f"Bootstrap OOB MSE: {best_model_info['Bootstrap_OOB_MSE_Mean']:.2f} ± {best_model_info['Bootstrap_OOB_MSE_Std']:.2f}")
    print(f"Final Test MSE: {best_model_info['Test_MSE']:.2f}")

    # Plot results
    plot_bootstrap_results(results_df)

    # Plot best model with uncertainty
    plot_best_model_with_uncertainty(
        X, y, best_model_info['Model'], best_model_info['Degree'],
        X_test, y_test, n_bootstrap=100
    )

    # Display results table
    print("\nDetailed Results:")
    display_df = results_df[['Degree', 'Bootstrap_OOB_MSE_Mean', 'Bootstrap_OOB_MSE_Std', 'Test_MSE']].round(2)
    display_df.columns = ['Degree', 'OOB_MSE_Mean', 'OOB_MSE_Std', 'Test_MSE']
    print(display_df.to_string(index=False))