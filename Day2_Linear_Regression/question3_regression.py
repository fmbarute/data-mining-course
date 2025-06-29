beta_0 = 50;
beta_1 = 20;
beta_2 = 0.07;
beta_3 = 35;
beta_4 = 0.01;
beta_5 = -10


def predict_salary(gpa, iq, level):
    return beta_0 + beta_1 * gpa + beta_2 * iq + beta_3 * level + beta_4 * (gpa * iq) + beta_5 * (gpa * level)


print("Linear Regression Exercise Solutions")
print("====================================")


print("\nPART (A): Education Level Comparison")


breakeven_gpa = 35 / 10
print(f"Breakeven GPA: {breakeven_gpa:.1f}")
print(f"College better when: GPA < {breakeven_gpa:.1f}")
print(f"High School better when: GPA > {breakeven_gpa:.1f}")

# Test examples
test_cases = [{'GPA': 3.0, 'IQ': 110}, {'GPA': 4.0, 'IQ': 110}]
for case in test_cases:
    hs = predict_salary(case['GPA'], case['IQ'], 0)
    college = predict_salary(case['GPA'], case['IQ'], 1)
    winner = "College" if college > hs else "High School"
    print(f"GPA {case['GPA']:.1f}: HS=${hs:.1f}k, College=${college:.1f}k -> {winner} wins")

print("ANSWER: (iii) High school graduates earn more when GPA is high enough")


print("\nPART (B): Salary Prediction")

gpa = 4.0;
iq = 110;
level = 1
predicted = predict_salary(gpa, iq, level)

print(f"College graduate with GPA={gpa:.1f}, IQ={iq}")
print(f"Calculation: 50 + 20*{gpa:.1f} + 0.07*{iq} + 35*{level} + 0.01*({gpa:.1f}*{iq}) - 10*({gpa:.1f}*{level})")
print(
    f"           = 50 + {20 * gpa:.1f} + {0.07 * iq:.1f} + {35 * level:.1f} + {0.01 * gpa * iq:.1f} - {10 * gpa * level:.1f}")
print(f"ANSWER: ${predicted:.1f} thousand")


print("\nPART (C): GPA/IQ Interaction Effect")

print("Statement: 'Small coefficient = little evidence of interaction'")
print("ANSWER: FALSE")

# Show practical impact
examples = [
    {'GPA': 3.0, 'IQ': 100},
    {'GPA': 4.0, 'IQ': 110},
    {'GPA': 4.0, 'IQ': 130}
]

print("Why FALSE - Practical Impact:")
for ex in examples:
    effect = beta_4 * ex['GPA'] * ex['IQ'] * 1000
    print(f"  GPA={ex['GPA']:.1f}, IQ={ex['IQ']}: Interaction adds ${effect:.0f} to salary")

print("Coefficient size alone doesn't determine significance!")
print("Need to consider: variable scales, standard errors, t-statistics")

print("\n=== EXERCISE COMPLETE ===")
