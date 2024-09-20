# Comprehensive Exploratory Data Analysis (EDA) in Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set visualization styles
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Loading the Data
# For demonstration, we'll use the 'tips' dataset from seaborn
df = sns.load_dataset('tips')
print("First 5 Rows of the Dataset:")
print(df.head())

# 2. Basic Information
print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# 3. Handling Missing Values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Visualize missing values
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# 4. Data Types and Categorical vs Numerical
print("\nData Types:")
print(df.dtypes)

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("\nCategorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)

# 5. Univariate Analysis

# 5.1 Numerical Features

# Histograms
df[numerical_cols].hist(bins=15, figsize=(15, 6), layout=(2, 2))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Box Plots
for col in numerical_cols:
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
    plt.show()

# Summary Statistics
print("\nDetailed Summary Statistics:")
for col in numerical_cols:
    print(f"\nStatistics for {col}:")
    print(f"Mean: {df[col].mean()}")
    print(f"Median: {df[col].median()}")
    print(f"Mode: {df[col].mode()[0]}")
    print(f"Variance: {df[col].var()}")
    print(f"Standard Deviation: {df[col].std()}")
    print(f"Skewness: {df[col].skew()}")
    print(f"Kurtosis: {df[col].kurtosis()}")

# 5.2 Categorical Features

# Count Plots
for col in categorical_cols:
    sns.countplot(x=df[col])
    plt.title(f'Count Plot of {col}')
    plt.show()

# Bar Plots with Aggregated Statistics
for col in categorical_cols:
    sns.barplot(x=col, y='total_bill', data=df, ci=None)
    plt.title(f'Average Total Bill by {col}')
    plt.show()

# 6. Bivariate Analysis

# 6.1 Numerical vs Numerical

# Scatter Plots
sns.scatterplot(x='total_bill', y='tip', data=df)
plt.title('Scatter Plot of Total Bill vs Tip')
plt.show()

# Correlation Matrix
corr_matrix = df.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Heatmap of Correlation Matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 6.2 Numerical vs Categorical

# Box Plot
sns.boxplot(x='day', y='total_bill', data=df)
plt.title('Total Bill Distribution by Day')
plt.show()

# Violin Plot
sns.violinplot(x='time', y='tip', data=df)
plt.title('Tip Distribution by Time')
plt.show()

# 6.3 Categorical vs Categorical

# Crosstab and Heatmap
crosstab = pd.crosstab(df['sex'], df['smoker'])
print("\nCrosstab between Sex and Smoker:")
print(crosstab)

sns.heatmap(crosstab, annot=True, fmt="d", cmap='Blues')
plt.title('Heatmap of Sex vs Smoker')
plt.show()

# 7. Multivariate Analysis

# Pair Plot
sns.pairplot(df, hue='smoker')
plt.suptitle('Pair Plot Colored by Smoker Status', y=1.02)
plt.show()

# 8. Outlier Detection

# Using Z-Score
z_scores = np.abs(stats.zscore(df[numerical_cols]))
print("\nZ-Scores:")
print(z_scores)

# Identify outliers (|z| > 3)
outliers = (z_scores > 3).any(axis=1)
print(f"\nNumber of Outliers: {outliers.sum()}")

# Box Plot to visualize outliers
sns.boxplot(data=df[numerical_cols])
plt.title('Box Plot for Outlier Detection')
plt.show()

# 9. Feature Relationships

# Heatmap with Pairwise Relationships
sns.pairplot(df, hue='sex', markers=["o", "s"])
plt.suptitle('Pairwise Relationships Colored by Sex', y=1.02)
plt.show()

# 10. Advanced Statistical Analysis

# 10.1 Pearson Correlation Coefficient
pearson_corr, pearson_p = stats.pearsonr(df['total_bill'], df['tip'])
print(f"\nPearson Correlation between Total Bill and Tip: {pearson_corr:.2f} (p-value: {pearson_p:.4f})")

# 10.2 Spearman Rank Correlation
spearman_corr, spearman_p = stats.spearmanr(df['total_bill'], df['tip'])
print(f"Spearman Correlation between Total Bill and Tip: {spearman_corr:.2f} (p-value: {spearman_p:.4f})")

# 10.3 T-Test: Comparing Tips between Smokers and Non-Smokers
tips_smokers = df[df['smoker'] == 'Yes']['tip']
tips_nonsmokers = df[df['smoker'] == 'No']['tip']
t_stat, t_p = stats.ttest_ind(tips_smokers, tips_nonsmokers)
print(f"\nT-Test between Smokers and Non-Smokers Tips: t-statistic = {t_stat:.2f}, p-value = {t_p:.4f}")

# 10.4 ANOVA: Comparing Total Bill across Days
anova = stats.f_oneway(df[df['day'] == 'Thur']['total_bill'],
                      df[df['day'] == 'Fri']['total_bill'],
                      df[df['day'] == 'Sat']['total_bill'],
                      df[df['day'] == 'Sun']['total_bill'])
print(f"\nANOVA for Total Bill across Days: F-statistic = {anova.statistic:.2f}, p-value = {anova.pvalue:.4f}")

# 11. Pivot Tables

# Pivot Table: Average Tip by Day and Time
pivot = pd.pivot_table(df, values='tip', index='day', columns='time', aggfunc='mean')
print("\nPivot Table - Average Tip by Day and Time:")
print(pivot)

sns.heatmap(pivot, annot=True, fmt=".2f", cmap='YlGnBu')
plt.title('Average Tip by Day and Time')
plt.show()

# 12. Conclusion
print("\n--- End of Exploratory Data Analysis ---")