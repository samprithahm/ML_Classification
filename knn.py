import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap

from sklearn import neighbors
hazelnut = pd.read_csv('./hazelnut.csv')
print("Top 5 Rows")
hazelnut.head()
hazelnut.shape

X = hazelnut.drop(['variety'],axis=1)
print(X.head())

y = hazelnut['variety'].values
print(y[:5])