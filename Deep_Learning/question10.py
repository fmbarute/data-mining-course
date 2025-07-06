from sklearn.linear_model import LinearRegression

# Create sequences for RNN-style approach
sequence_length = 5
sequences = []
targets = []
for i in range(len(nyse) - sequence_length):
    sequences.append(nyse.iloc[i:i+sequence_length]['Return'].values)
    targets.append(nyse.iloc[i+sequence_length]['Return'])
sequences = np.array(sequences)
targets = np.array(targets)

# Split into train/test
split_idx = int(0.8 * len(sequences))
X_seq_train, X_seq_test = sequences[:split_idx], sequences[split_idx:]
y_seq_train, y_seq_test = targets[:split_idx], targets[split_idx:]

# Flatten sequences for linear model
X_flat_train = X_seq_train.reshape(X_seq_train.shape[0], -1)
X_flat_test = X_seq_test.reshape(X_seq_test.shape[0], -1)

# Fit linear model
flat_model = LinearRegression()
flat_model.fit(X_flat_train, y_seq_train)
flat_r2 = r2_score(y_seq_test, flat_model.predict(X_flat_test))

print(f"Flattened sequence model R2: {flat_r2:.4f}")
print(f"Traditional AR model R2: {r2_no_month:.4f}")

# Advantages/disadvantages:
print("""
Advantages of flattened approach:
- Easier to incorporate more complex feature engineering
- Can handle variable sequence lengths more flexibly

Disadvantages:
- Loses the explicit time ordering structure
- May require more data to achieve same performance
- Harder to interpret coefficients
""")