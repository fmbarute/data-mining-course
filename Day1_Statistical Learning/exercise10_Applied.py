from ISLP import load_data


def analyze_boston():
    boston = load_data('Boston')
    print(f"Dataset shape: {boston.shape}")
    # Add other analyses
