import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("🎓 College Dataset Statistical Learning ")
print("="*60)

print("\n(a) Reading data from ISLR website...")

url = "https://www.statlearning.com/s/College.csv"
college = pd.read_csv(url)
print(f"✅ Data downloaded. Shape: {college.shape}")
print("First few rows:")
print(college.head())


print("\n(b) Handling university names as index...")

# Method 1: Read with index_col parameter (cleaner approach)
college2 = pd.read_csv(url, index_col=0)
print("college2 shape:", college2.shape)

college3 = college.rename({'Unnamed: 0': 'College'}, axis=1)
college3 = college3.set_index('College')
print("college3 shape:", college3.shape)

college = college3
print(f"✅ Final dataset shape: {college.shape}")
print("Index name:", college.index.name)
print("First few universities:", list(college.index[:5]))


print("\n(c) Descriptive statistics...")

desc_stats = college.describe()
print(desc_stats)

# Additional info
print(f"\nPrivate vs Public count:")
print(college['Private'].value_counts())


print("\n(d) Creating scatter matrix...")

# Create scatter matrix for specified columns
columns_to_plot = ['Top10perc', 'Apps', 'Enroll']
pd.plotting.scatter_matrix(college[columns_to_plot], figsize=(10, 8), alpha=0.6)
plt.suptitle('Scatter Matrix: Top10perc, Apps, Enroll')
plt.tight_layout()
plt.show()

# Show correlations between these variables
print("Correlations:")
print(college[columns_to_plot].corr().round(3))


print("\n(e) Boxplot: Outstate vs Private...")

college.boxplot(column='Outstate', by='Private', figsize=(8, 6))
plt.title('Out-of-State Tuition by University Type')
plt.xlabel('University Type (Private: No=Public, Yes=Private)')
plt.ylabel('Out-of-State Tuition ($)')
plt.suptitle('')
plt.show()

# Summary statistics by group
print("Outstate tuition by university type:")
print(college.groupby('Private')['Outstate'].describe())


print("\n(f) Creating Elite variable...")

# Create Elite variable using pd.cut() - bin Top10perc into two groups
college['Elite'] = pd.cut(college['Top10perc'], bins=[0, 50, 100], labels=['No', 'Yes'])
# pd.cut bins Top10perc: 0-50% = 'No', 50-100% = 'Yes'

# Count elite universities
elite_counts = college['Elite'].value_counts()  # Count how many in each Elite category
print("Elite university counts:")
print(elite_counts)  # Display the counts
print(f"Percentage elite: {elite_counts['Yes']/len(college)*100:.1f}%")  # Calculate percentage

# Boxplot Outstate vs Elite
college.boxplot(column='Outstate', by='Elite', figsize=(8, 6))
plt.title('Out-of-State Tuition by Elite Status')
plt.xlabel('Elite Status (>50% from top 10% of high school)')
plt.ylabel('Out-of-State Tuition ($)')
plt.suptitle('')
plt.show()


print("\n(g) Creating histograms with different bin numbers...")

# Create 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create 2x2 grid of subplots
axes = axes.ravel()  # Flatten 2D array to 1D for easier indexing

# Variables to plot and their bin counts
variables = ['Apps', 'Enroll', 'Expend', 'Grad.Rate']  # List of variables to plot
bin_counts = [15, 25, 30, 20]  # Different bin numbers for each histogram

# Create histogram for each variable
for i, (var, bins) in enumerate(zip(variables, bin_counts)):  # Loop through variables and bin counts
    axes[i].hist(college[var], bins=bins, alpha=0.7, edgecolor='black')  # Create histogram
    axes[i].set_title(f'{var} ({bins} bins)')  # Set title with variable name and bin count
    axes[i].set_xlabel(var)  # Label x-axis with variable name
    axes[i].set_ylabel('Frequency')  # Label y-axis

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Display all histograms

# ============================================================================
# Part (h): Additional exploration
# ============================================================================
print("\n(h) Additional data exploration...")

# 1. Create new derived variables
college['Accept_Rate'] = (college['Accept'] / college['Apps']) * 100  # Calculate acceptance rate percentage
college['Total_Students'] = college['F.Undergrad'] + college['P.Undergrad']  # Total undergraduate students
college['Total_Cost'] = college['Outstate'] + college['Room.Board'] + college['Books'] + college['Personal']  # Total cost

print("New variables created:")
print("• Accept_Rate: Acceptance rate percentage")
print("• Total_Students: Total undergraduate enrollment")
print("• Total_Cost: Total estimated annual cost")

# 2. Find strongest correlations
numeric_cols = college.select_dtypes(include=[np.number]).columns  # Get all numeric columns
corr_matrix = college[numeric_cols].corr()  # Calculate correlation matrix

# Find pairs with highest correlations (excluding self-correlations)
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append({
            'var1': corr_matrix.columns[i],
            'var2': corr_matrix.columns[j],
            'correlation': corr_matrix.iloc[i, j]
        })

corr_df = pd.DataFrame(corr_pairs)  # Convert to DataFrame
top_corr = corr_df.reindex(corr_df['correlation'].abs().sort_values(ascending=False).index)

print(f"\nTop 5 strongest correlations:")
for _, row in top_corr.head(5).iterrows():  # Loop through top 5 correlations
    print(f"• {row['var1']} vs {row['var2']}: {row['correlation']:.3f}")  # Display correlation

# 3. University size analysis
college['Size'] = pd.cut(college['Total_Students'], bins=[0, 2000, 10000, float('inf')],
                        labels=['Small', 'Medium', 'Large'])  # Categorize by size

print(f"\nUniversity size distribution:")
print(college['Size'].value_counts())  # Count universities by size

# 4. Quality comparison
print(f"\nElite vs Non-Elite comparison (key metrics):")
elite_comparison = college.groupby('Elite')[['Accept_Rate', 'Grad.Rate', 'Expend', 'PhD']].mean()  # Compare groups
print(elite_comparison.round(2))  # Display rounded means

# 5. Final comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # Create 2x2 plot grid

# Plot 1: Apps vs Accept Rate
axes[0,0].scatter(college['Apps'], college['Accept_Rate'], alpha=0.6)  # Scatter plot
axes[0,0].set_xlabel('Number of Applications')  # Label x-axis
axes[0,0].set_ylabel('Acceptance Rate (%)')  # Label y-axis
axes[0,0].set_title('Applications vs Acceptance Rate')  # Set title

# Plot 2: Expenditure vs Graduation Rate
axes[0,1].scatter(college['Expend'], college['Grad.Rate'], alpha=0.6)  # Scatter plot
axes[0,1].set_xlabel('Expenditure per Student ($)')  # Label x-axis
axes[0,1].set_ylabel('Graduation Rate (%)')  # Label y-axis
axes[0,1].set_title('Expenditure vs Graduation Rate')  # Set title

# Plot 3: Private vs Public average tuition
tuition_avg = college.groupby('Private')['Outstate'].mean()  # Calculate average tuition by type
axes[1,0].bar(tuition_avg.index, tuition_avg.values, alpha=0.7)  # Bar plot
axes[1,0].set_xlabel('University Type')  # Label x-axis
axes[1,0].set_ylabel('Average Out-of-State Tuition ($)')  # Label y-axis
axes[1,0].set_title('Average Tuition by Type')  # Set title

# Plot 4: Alumni donations vs Graduation Rate
axes[1,1].scatter(college['perc.alumni'], college['Grad.Rate'], alpha=0.6)  # Scatter plot
axes[1,1].set_xlabel('Alumni Donation Percentage')  # Label x-axis
axes[1,1].set_ylabel('Graduation Rate (%)')  # Label y-axis
axes[1,1].set_title('Alumni Donations vs Graduation Rate')

plt.tight_layout()
plt.show()


print("\n" + "="*60)
print("📋 KEY DISCOVERIES")
print("="*60)

print(f"1. DATASET OVERVIEW:")
print(f"   • Total universities: {len(college)}")  # Total number of universities
print(f"   • Variables analyzed: {len(college.columns)}")  # Number of variables
print(f"   • Private universities: {(college['Private'] == 'Yes').sum()} ({(college['Private'] == 'Yes').sum()/len(college)*100:.1f}%)")

print(f"\n2. TUITION DIFFERENCES:")
private_avg = college[college['Private'] == 'Yes']['Outstate'].mean()  # Average private tuition
public_avg = college[college['Private'] == 'No']['Outstate'].mean()  # Average public tuition
print(f"   • Private university average tuition: ${private_avg:,.0f}")
print(f"   • Public university average tuition: ${public_avg:,.0f}")
print(f"   • Private costs {private_avg/public_avg:.1f}x more than public")

print(f"\n3. ELITE UNIVERSITIES:")
elite_pct = (college['Elite'] == 'Yes').sum() / len(college) * 100  # Calculate elite percentage
print(f"   • Elite universities (>50% top 10% students): {elite_pct:.1f}%")
elite_grad = college[college['Elite'] == 'Yes']['Grad.Rate'].mean()  # Elite graduation rate
non_elite_grad = college[college['Elite'] == 'No']['Grad.Rate'].mean()  # Non-elite graduation rate
print(f"   • Elite avg graduation rate: {elite_grad:.1f}%")
print(f"   • Non-elite avg graduation rate: {non_elite_grad:.1f}%")

print(f"\n4. STRONGEST CORRELATION:")
strongest = top_corr.iloc[0]  # Get strongest correlation
print(f"   • {strongest['var1']} vs {strongest['var2']}: {strongest['correlation']:.3f}")

print(f"\n5. ACCEPTANCE PATTERNS:")
print(f"   • Average acceptance rate: {college['Accept_Rate'].mean():.1f}%")
print(f"   • Range: {college['Accept_Rate'].min():.1f}% to {college['Accept_Rate'].max():.1f}%")

print(f"\n✅ Analysis Complete!")
print(f"📊 All parts (a) through (h) successfully executed with minimal code")