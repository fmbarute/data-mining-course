def classification_apps():
    apps = [
        {
            'application': 'Spam Detection',
            'response': 'Spam/Not Spam',
            'predictors': 'Email content, sender, subject',
            'goal': 'Prediction'
        },
        # Add 2 more
    ]
    print(pd.DataFrame(apps).to_markdown())