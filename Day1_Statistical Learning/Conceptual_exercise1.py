def flexibility_analysis():
    answers = {
        'a': 'Better - With large n, flexible methods can model complex patterns without overfitting',
        'b': 'Worse - High p with small n causes overfitting (curse of dimensionality)',
        'c': 'Better - Flexible methods capture non-linear relationships',
        'd': 'Worse - High error variance requires simpler models to avoid fitting noise'
    }
    for q, ans in answers.items():
        print(f"({q}) {ans}")


if __name__ == "__main__":
    flexibility_analysis()
