import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('fake-reviews.csv')

# Add features for EDA
df['char_length'] = df['text_'].apply(len)  # Character length of each review
df['word_count'] = df['text_'].str.split().apply(len)  # Word count of each review

# Exploratory Data Analysis (EDA)

print("\nDataFrame Info:")
df.info()

print("\nDataFrame Describe:")
print(df.describe(include='all'))

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nUnique Values in Each Column:")
print(df.nunique())

print("\nTop and Bottom 5 Rows of DataFrame sorted by 'rating':")
print(df.sort_values(by='rating', ascending=False).head())
print(df.sort_values(by='rating', ascending=False).tail())

print("\nFrequency of rating:")
print(df['rating'].value_counts())


# Correlation matrix heatmap (needs more numerical features to make sense)
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()
print('test')

# Combined boxplots: character length and word count

# --- Boxplots (currently commented out) ---
# Boxplots are useful for visualizing the distribution of numerical data.
# They show the median, quartiles (25th and 75th percentiles), and potential outliers.
# The box represents the interquartile range (IQR), the line inside the box is the median,
# and the whiskers extend to show the rest of the distribution except for outliers, which are plotted as individual points.
#
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
#
# sns.boxplot(x=df['char_length'], ax=axes[0], color='skyblue')
# axes[0].set_title('Boxplot of Character Lengths')
# axes[0].set_xlabel('Character Length')
#
# sns.boxplot(x=df['word_count'], ax=axes[1], color='lightgreen')
# axes[1].set_title('Boxplot of Word Counts')
# axes[1].set_xlabel('Word Count')
#
# plt.tight_layout()
# plt.show()