import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np





# Load the dataset
california = fetch_california_housing(as_frame=True)
df = california.frame
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

#Plot histograms of all features
df.hist(bins=50, figsize=(15, 10))
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

#Plot pairplot of all features
sns.pairplot(df.sample(frac=0.1, random_state=42), diag_kind="kde")
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

#Plot correlation matrix of all features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

#IQR ile aykırı değer tespiti
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("Özellik bazında aykırı değer sayıları:\n", outliers)