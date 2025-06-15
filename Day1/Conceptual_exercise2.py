import pandas as pd


def problem_classification():
    scenarios = [
        {
            'type': 'Regression',
            'goal': 'Inference',
            'n': 500,
            'p': 3  # profit, employees, industry
        },
        # Add other scenarios similarly
    ]

    df = pd.DataFrame(scenarios)
    print(df.to_markdown())


if __name__ == "__main__":
    problem_classification()