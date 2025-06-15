def analyze_auto():
    auto = pd.read_csv('Auto.csv').dropna()
    print(f"Quantitative predictors: {auto.select_dtypes(include='number').columns.tolist()}")
    # Add other analyses