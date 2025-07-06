import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ISLP import load_data

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("="*60)
print("CHAPTER 2, QUESTION 8: COLLEGE DATASET ANALYSIS")
print("="*60)

# Part (a): Load the College data
print("\n(a) Loading College dataset...")
college = load_data('College')
print(f"Dataset loaded successfully!")
print(f"Shape: {college.shape}")

# Part (b): Examine the data structure
print("\n(b) Examining data structure...")
print("First few rows:")
print(college.head())
print(f"\nDataset info:")
print(college.info())

# The index should be college names - let's verify
print(f"\nIndex (college names) sample:")
print(college.index[:5].tolist())

# Part (c): Numerical summary
print("\n(c) Numerical summary of all variables...")
summary_stats = college.describe()
print(summary_stats)

# Save key insights
print(f"\nKey insights from summary:")
print(f"- Number of colleges: {len(college)}")
print(f"- Acceptance rates range from {college['Accept'].min()/college['Apps'].max()*100:.1f}% to {college['Accept'].max()/college['Apps'].min()*100:.1f}%")
print(f"- Graduation rates range from {college['Grad.Rate'].min():.1f}% to {college['Grad.Rate'].max():.1f}%")

# Part (d): Scatterplot matrix
print("\n(d) Creating scatterplot matrix...")
selected_cols = ['Top10perc', 'Apps', 'Enroll']
fig, axes = plt.subplots(figsize=(12, 8))
pd.plotting.scatter_matrix(college[selected_cols],
                          alpha=0.6,
                          figsize=(12, 8),
                          diagonal='hist')
plt.suptitle('Scatterplot Matrix: Top10perc, Apps, Enroll', fontsize=14)
plt.tight_layout()
plt.show()

# Part (e): Boxplot of Outstate vs Private
print("\n(e) Boxplot: Outstate tuition by Private/Public...")
plt.figure(figsize=(10, 6))
college.boxplot(column='Outstate', by='Private', ax=plt.gca())
plt.title('Out-of-State Tuition by College Type')
plt.xlabel('Private College (No/Yes)')
plt.ylabel('Out-of-State Tuition ($)')
plt.show()

# Statistical summary for the boxplot
print("Outstate tuition summary by college type:")
outstate_summary = college.groupby('Private')['Outstate'].describe()
print(outstate_summary)

# Part (f): Create Elite variable and analyze
print("\n(f) Creating Elite variable...")
# Create Elite variable: colleges with >50% students from top 10% of HS class
college['Elite'] = pd.cut(college['Top10perc'],
                         bins=[0, 50, 100],
                         labels=['No', 'Yes'])

# Count elite universities
elite_counts = college['Elite'].value_counts()
print(f"Elite university counts:")
print(elite_counts)
print(f"Percentage of elite universities: {elite_counts['Yes']/len(college)*100:.1f}%")

# Boxplot: Outstate vs Elite
plt.figure(figsize=(10, 6))
college.boxplot(column='Outstate', by='Elite', ax=plt.gca())
plt.title('Out-of-State Tuition by Elite Status')
plt.xlabel('Elite College (>50% from top 10% of HS)')
plt.ylabel('Out-of-State Tuition ($)')
plt.show()

# Part (g): Histograms with different bins
print("\n(g) Creating histograms with different bin sizes...")
quantitative_vars = ['Apps', 'Accept', 'Enroll', 'Top10perc']
bin_sizes = [10, 20, 30, 50]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, var in enumerate(quantitative_vars):
    axes[i].hist(college[var], bins=bin_sizes[i], alpha=0.7, edgecolor='black')
    axes[i].set_title(f'{var} (bins={bin_sizes[i]})')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Part (h): Additional exploration
print("\n(h) Additional data exploration...")

# 1. Correlation analysis
print("1. Correlation between key variables:")
corr_vars = ['Apps', 'Accept', 'Enroll', 'Top10perc', 'Outstate', 'Grad.Rate']
correlation_matrix = college[corr_vars].corr()
print(correlation_matrix.round(3))

# Heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Key College Variables')
plt.tight_layout()
plt.show()

# 2. Most selective colleges
print("\n2. Most selective colleges (lowest acceptance rate):")
college['Accept_Rate'] = college['Accept'] / college['Apps'] * 100
most_selective = college.nsmallest(10, 'Accept_Rate')[['Apps', 'Accept', 'Accept_Rate', 'Private']]
print(most_selective)

# 3. Relationship between room/board costs and private status
print("\n3. Room & Board costs by college type:")
rb_summary = college.groupby('Private')['Room.Board'].describe()
print(rb_summary)

# 4. Distribution of student-faculty ratio
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(college['S.F.Ratio'], bins=20, alpha=0.7, edgecolor='black')
plt.title('Distribution of Student-Faculty Ratio')
plt.xlabel('Student-Faculty Ratio')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
college.boxplot(column='S.F.Ratio', by='Private', ax=plt.gca())
plt.title('Student-Faculty Ratio by College Type')
plt.xlabel('Private College (No/Yes)')
plt.ylabel('Student-Faculty Ratio')
plt.tight_layout()
plt.show()

# 5. Interesting findings
print("\n5. Key findings from exploration:")
print(f"- Private colleges have higher average out-of-state tuition: ${college.groupby('Private')['Outstate'].mean()['Yes']:.0f} vs ${college.groupby('Private')['Outstate'].mean()['No']:.0f}")
print(f"- Elite colleges have higher graduation rates: {college.groupby('Elite')['Grad.Rate'].mean()['Yes']:.1f}% vs {college.groupby('Elite')['Grad.Rate'].mean()['No']:.1f}%")
print(f"- Strong correlation between Apps and Accept: {college[['Apps', 'Accept']].corr().iloc[0,1]:.3f}")
print(f"- {len(college[college['Grad.Rate'] > 100])} colleges have graduation rates > 100% (data quality issue)")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)