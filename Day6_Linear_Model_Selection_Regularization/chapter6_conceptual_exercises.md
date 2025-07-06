# Chapter 6: Linear Model Selection and Regularization - Conceptual Exercises

## Exercise 1: Model Selection Methods Comparison

### (a) Which model with k predictors has the smallest training RSS?

**Answer: Best Subset Selection**

**Justification:** Best subset selection considers all possible combinations of k predictors and chooses the one that minimizes training RSS. Forward and backward stepwise are greedy algorithms that make locally optimal choices but may miss the globally optimal combination.

### (b) Which model with k predictors has the smallest test RSS?

**Answer: Any of the three could have the smallest test RSS**

**Justification:** Test RSS depends on how well the model generalizes. While best subset finds the optimal training fit, it may overfit. Forward/backward stepwise might find simpler, more generalizable models. The answer depends on the specific dataset and the bias-variance tradeoff.

### (c) True or False Statements:

**i. Forward stepwise k-variable ⊆ Forward stepwise (k+1)-variable**
- **TRUE** - Forward stepwise adds one predictor at each step, so the k-variable model is always contained in the (k+1)-variable model.

**ii. Backward stepwise k-variable ⊆ Backward stepwise (k+1)-variable**  
- **TRUE** - Backward stepwise removes one predictor at each step, so the k-variable model is always contained in the (k+1)-variable model.

**iii. Backward stepwise k-variable ⊆ Forward stepwise (k+1)-variable**
- **FALSE** - These are independent procedures that may select completely different sets of predictors.

**iv. Forward stepwise k-variable ⊆ Backward stepwise (k+1)-variable**
- **FALSE** - These are independent procedures that may select completely different sets of predictors.

**v. Best subset k-variable ⊆ Best subset (k+1)-variable**
- **FALSE** - Best subset selection chooses the optimal set for each size independently. Adding one more predictor might result in a completely different optimal set.

---

## Exercise 2: Lasso vs Least Squares Flexibility

### (a) The lasso, relative to least squares, is:

**Answer: iii. Less flexible and hence will give improved prediction accuracy when its increase in bias is less than its decrease in variance.**

**Justification:** 
- Lasso adds an L1 penalty that constrains coefficients, making it less flexible than OLS
- This constraint increases bias but reduces variance
- Prediction improves when the variance reduction outweighs the bias increase

### (b) Ridge regression relative to least squares:

**Answer: iii. Less flexible and hence will give improved prediction accuracy when its increase in bias is less than its decrease in variance.**

**Justification:**
- Ridge adds an L2 penalty that shrinks coefficients toward zero
- Like lasso, this makes the model less flexible, increasing bias but reducing variance
- Performance improves when variance reduction dominates bias increase

### (c) Non-linear methods relative to least squares:

**Answer: ii. More flexible and hence will give improved prediction accuracy when its increase in variance is less than its decrease in bias.**

**Justification:**
- Non-linear methods (polynomials, splines, etc.) can capture complex relationships
- They are more flexible than linear methods, decreasing bias but increasing variance
- They improve prediction when the bias reduction outweighs the variance increase

---

## Exercise 3: Lasso Constraint Analysis (s parameter)

As we increase s from 0:

### (a) Training RSS: **iv. Steadily decrease**
- At s=0: All βⱼ=0, worst possible fit
- As s increases: More flexibility to fit training data
- Training RSS can only improve or stay the same

### (b) Test RSS: **ii. Decrease initially, then eventually start increasing in a U shape**
- Small s: High bias, low variance (underfitting)
- Medium s: Optimal bias-variance tradeoff
- Large s: Low bias, high variance (overfitting)
- Classic U-shaped test error curve

### (c) Variance: **iii. Steadily increase**
- More flexibility → more sensitivity to training data
- Variance increases monotonically with model complexity

### (d) (Squared) Bias: **iv. Steadily decrease**
- More flexibility → better approximation of true function
- Bias decreases as model approaches true relationship

### (e) Irreducible Error: **v. Remain constant**
- Inherent noise in data, independent of model choice
- No model can reduce irreducible error

---

## Exercise 4: Ridge Regression Analysis (λ parameter)

As we increase λ from 0:

### (a) Training RSS: **iii. Steadily increase**
- λ=0: Equivalent to OLS (best training fit)
- Higher λ: More constraint → worse training fit
- Training RSS increases monotonically

### (b) Test RSS: **ii. Decrease initially, then eventually start increasing in a U shape**
- Small λ: May overfit (high variance)
- Medium λ: Optimal bias-variance balance
- Large λ: Underfit (high bias)
- U-shaped test error pattern

### (c) Variance: **iv. Steadily decrease**
- More regularization → less sensitivity to training data
- Variance decreases as coefficients shrink

### (d) (Squared) Bias: **iii. Steadily increase**
- More constraint → poorer approximation of true function
- Bias increases with regularization strength

### (e) Irreducible Error: **v. Remain constant**
- Independent of model choice
- Represents inherent data noise

---

## Exercise 5: Ridge vs Lasso with Correlated Variables

Given: n=2, p=2, x₁₁=x₁₂, x₂₁=x₂₂, y₁+y₂=0, x₁₁+x₂₁=0, x₁₂+x₂₂=0

### (a) Ridge regression optimization:
Minimize: (y₁ - β₁x₁₁ - β₂x₁₂)² + (y₂ - β₁x₂₁ - β₂x₂₂)² + λ(β₁² + β₂²)

Since x₁₁=x₁₂ and x₂₁=x₂₂:
Minimize: (y₁ - β₁x₁₁ - β₂x₁₁)² + (y₂ - β₁x₂₁ - β₂x₂₁)² + λ(β₁² + β₂²)
= (y₁ - (β₁ + β₂)x₁₁)² + (y₂ - (β₁ + β₂)x₂₁)² + λ(β₁² + β₂²)

### (b) Ridge coefficient equality:
Since x₁₁+x₂₁=0, we have x₂₁=-x₁₁. The objective becomes:
(y₁ - (β₁ + β₂)x₁₁)² + (y₂ + (β₁ + β₂)x₁₁)² + λ(β₁² + β₂²)

For this to be minimized, the function is symmetric in β₁ and β₂, leading to β̂₁ = β̂₂.

### (c) Lasso optimization:
Minimize: (y₁ - β₁x₁₁ - β₂x₁₂)² + (y₂ - β₁x₂₁ - β₂x₂₂)² + λ(|β₁| + |β₂|)

### (d) Lasso non-uniqueness:
The lasso objective can be rewritten as:
(y₁ - (β₁ + β₂)x₁₁)² + (y₂ + (β₁ + β₂)x₁₁)² + λ(|β₁| + |β₂|)

This depends only on (β₁ + β₂), not on β₁ and β₂ individually. Any combination of β₁ and β₂ that gives the same sum will yield the same objective value, making the solution non-unique.

---

## Exercise 6: Soft Thresholding Visualization

### (a) Ridge (L2) case with p=1:
For the objective: ½(y₁ - β₁)² + λβ₁²

Taking derivative and setting to 0:
-(y₁ - β₁) + 2λβ₁ = 0
β₁ = y₁/(1 + 2λ)

**Plot characteristics:**
- Smooth, differentiable function
- Unique minimum
- Solution shrinks toward zero as λ increases

### (b) Lasso (L1) case with p=1:
For the objective: ½(y₁ - β₁)² + λ|β₁|

**Solution (soft thresholding):**
- If y₁ > λ: β₁ = y₁ - λ
- If y₁ < -λ: β₁ = y₁ + λ  
- If |y₁| ≤ λ: β₁ = 0

**Plot characteristics:**
- V-shaped penalty creates kink at β₁=0
- Non-differentiable at zero
- Exact zero solution for small |y₁|

---

## Exercise 7: Bayesian Connection

### (a) Likelihood for normal errors:
L(β₀, β₁,..., βₚ, σ²) = ∏ᵢ₌₁ⁿ (1/√(2πσ²)) exp(-(yᵢ - β₀ - Σⱼ₌₁ᵖ xᵢⱼβⱼ)²/(2σ²))

### (b) Posterior with double-exponential prior:
Prior: p(βⱼ) = (1/2b)exp(-|βⱼ|/b)

Posterior ∝ Likelihood × Prior
∝ exp(-1/(2σ²) Σᵢ(yᵢ - β₀ - Σⱼxᵢⱼβⱼ)²) × ∏ⱼexp(-|βⱼ|/b)
∝ exp(-1/(2σ²) Σᵢ(yᵢ - β₀ - Σⱼxᵢⱼβⱼ)² - (1/b)Σⱼ|βⱼ|)

### (c) Lasso connection:
The mode of this posterior is found by minimizing:
1/(2σ²) Σᵢ(yᵢ - β₀ - Σⱼxᵢⱼβⱼ)² + (1/b)Σⱼ|βⱼ|

This is equivalent to the lasso objective with λ = σ²/b.

### (d) Posterior with normal prior:
Prior: p(βⱼ) ~ N(0, c)

Posterior ∝ exp(-1/(2σ²) Σᵢ(yᵢ - β₀ - Σⱼxᵢⱼβⱼ)² - 1/(2c)Σⱼβⱼ²)

### (e) Ridge connection:
Both the mode and mean of this posterior minimize:
1/(2σ²) Σᵢ(yᵢ - β₀ - Σⱼxᵢⱼβⱼ)² + 1/(2c)Σⱼβⱼ²

This corresponds to ridge regression with λ = σ²/c.

---

## Summary

These exercises demonstrate:
1. **Model Selection**: Best subset is optimal for training but not necessarily test performance
2. **Regularization**: Both lasso and ridge reduce flexibility, trading bias for variance
3. **Parameter Effects**: s (lasso) and λ (ridge) control the bias-variance tradeoff
4. **Correlated Variables**: Ridge gives similar coefficients, lasso may not have unique solutions
5. **Bayesian Interpretation**: Regularization corresponds to different prior distributions