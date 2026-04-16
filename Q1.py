import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np





# Load the dataset
california = fetch_california_housing()
df = california.frame
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']


