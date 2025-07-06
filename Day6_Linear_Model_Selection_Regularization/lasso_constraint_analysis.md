# Question 3: Lasso Regression Constraint Analysis

**Problem Statement:** We minimize the regression objective function:
$$\sum_{i=1}^n \left( y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij} \right)^2$$

Subject to the constraint: $$\sum_{j=1}^p |\beta_j| \leq s$$

## Analysis of Parameter s

The constraint parameter s controls the complexity of the model:
- **When s = 0**: All coefficients βⱼ = 0 (only intercept β₀ allowed)
- **As s increases**: More coefficients can be non-zero and larger in magnitude
- **When s → ∞**: Approaches ordinary least squares (OLS) solution

---

## (a) Training RSS as s increases from 0

**Answer: iv. Steadily decrease**

**Justification:**
- At s = 0: Only β₀ is non-zero, giving the worst possible fit (highest RSS)
- As s increases: More flexibility to fit training data better
- The model can always achieve at least as good a fit as the previous constraint level
- Training RSS monotonically decreases toward the OLS solution

---

## (b) Test RSS as s increases from 0  

**Answer: ii. Decrease initially, and then eventually start increasing in a U shape**

**Justification:**
- **Initially (small s)**: Model is underfit with high bias, low variance
- **Middle range**: Bias decreases faster than variance increases → test RSS decreases
- **Large s**: Model becomes overfit with low bias but high variance → test RSS increases
- This creates the classic bias-variance tradeoff U-shaped curve

---

## (c) Variance as s increases from 0

**Answer: iii. Steadily increase**

**Justification:**
- At s = 0: Very simple model (intercept only) → minimal variance
- As s increases: Model becomes more complex and flexible
- More parameters can vary with different training samples
- Variance monotonically increases as model complexity grows

---

## (d) (Squared) Bias as s increases from 0

**Answer: iv. Steadily decrease**  

**Justification:**
- At s = 0: Severe underfitting → high bias (can't capture true relationship)
- As s increases: Model gains flexibility to approximate true function better
- Bias steadily decreases as model approaches the true underlying relationship
- At s → ∞: Approaches unbiased OLS estimator (for linear relationships)

---

## (e) Irreducible Error as s increases from 0

**Answer: v. Remain constant**

**Justification:**
- Irreducible error represents inherent noise in the data (ε in y = f(x) + ε)
- This noise is independent of the model choice or complexity
- No amount of model flexibility can reduce irreducible error
- It remains constant regardless of the constraint parameter s

---

## Summary: Bias-Variance Decomposition

As s increases from 0:

| Component | Behavior | Reason |
|-----------|----------|---------|
| **Training RSS** | ↓ Decreases | Better fit to training data |
| **Test RSS** | ↓ then ↑ U-shape | Bias-variance tradeoff |
| **Variance** | ↑ Increases | Model complexity increases |
| **Bias²** | ↓ Decreases | Model flexibility increases |
| **Irreducible Error** | → Constant | Independent of model |

The optimal value of s balances bias and variance to minimize test RSS.