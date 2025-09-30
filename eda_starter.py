import pandas as pd

# Load your dataset (make sure the path is correct)

df = pd.read_csv("fake-reviews.csv")

# Pandas is a library used for data manipulation and analysis

# Quick look at the first rows

print(df.head())

# Count missing values in each column

print(df.isnull().sum())

# Character length of each review


df["char_length"] = df["text_"].apply(len)

# Word count of each review

df["word_count"] = df["text_"].str.split().apply(len)

# The two lines above add 2 new categories to our data: char_length and word_count

import seaborn as sns # To visualize data

import matplotlib.pyplot as plt # To manipulate graphics of data

sns.boxplot(x="char_length", y="label", data=df)

plt.title("Character Length of Reviews by Label")

plt.xlabel("Character length")

plt.ylabel("Review label")

plt.show()