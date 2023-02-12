from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

diabetes = pd.read_csv('diabetes.csv')
X = diabetes.values[:, 0:8]
Y = diabetes.values[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))