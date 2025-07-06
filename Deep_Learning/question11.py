from sklearn.neural_network import MLPRegressor

# Using the same flattened sequences from Q10

# Fit nonlinear model
nonlinear_model = MLPRegressor(hidden_layer_sizes=(32,), activation='relu',
                              max_iter=500, random_state=42)
nonlinear_model.fit(X_flat_train, y_seq_train)
nonlinear_r2 = r2_score(y_seq_test, nonlinear_model.predict(X_flat_test))

print(f"Nonlinear AR model R2: {nonlinear_r2:.4f}")
print(f"Linear AR model R2: {flat_r2:.4f}")
print(f"Improvement: {nonlinear_r2 - flat_r2:.4f}")