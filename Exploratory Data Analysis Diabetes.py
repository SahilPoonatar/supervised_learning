import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

diabetes = pd.read_csv("diabetes.csv")
# print(diabetes.head())
# print(diabetes.shape)
# print(diabetes.dtypes)
# print(diabetes.describe())
# The first thing we notice is that the column of Outcome is binary, either 0 or 1
# There are 8 predictors, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction and Age
# There are 768 total rows

# The first thing we are going to check is to check the total number of positive outcomes (1) and the number of negative outcomes (0)
# number_of_occurences = diabetes["Outcome"].value_counts()
# print(number_of_occurences)
# We can see that that the number of positive outcomes is 268 and the number of negative outcomes is 500, nearly double of the positive outcomes
# The percentage of positive outcomes is only approximately 26%, which suggests that the dataset is imbalanced
# Let's check if there are any missing values
# print(diabetes.isnull().any())
# There aren't any

# Now let us begin with the EDA
# UNIVARIATE ANALYSIS
# Here, the daya being analyzed will belong only to onw variable.
# This type of analysis won't give us insight on relationshsips between variables.
# We will however be able to better describe the data.

# Univariate Box Plot
# plt.subplots(figsize=(15, 10))
#
# plt.subplot(2,4,1)
# boxplot = diabetes["Pregnancies"].plot(kind='box', title='boxplot')
#
# plt.subplot(2,4,2)
# boxplot = diabetes["Glucose"].plot(kind='box', title='boxplot')
#
# plt.subplot(2,4,3)
# boxplot = diabetes["BloodPressure"].plot(kind='box', title='boxplot')
#
# plt.subplot(2,4,4)
# boxplot = diabetes["SkinThickness"].plot(kind='box', title='boxplot')
#
# plt.subplot(2,4,5)
# boxplot = diabetes["Insulin"].plot(kind='box', title='boxplot')
#
# plt.subplot(2,4,6)
# boxplot = diabetes["BMI"].plot(kind='box', title='boxplot')
#
# plt.subplot(2,4,7)
# boxplot = diabetes["DiabetesPedigreeFunction"].plot(kind='box', title='boxplot')
#
# plt.subplot(2,4,8)
# boxplot = diabetes["Age"].plot(kind='box', title='boxplot')
#
# plt.show()

# From this we can see that the distribution of Insulin, Glucose and Skin Thickness are comparatively more spread out
# Whereas Blood Pressure, Age, Pregnancies and BMI are concentrated around its median.
# There are also an enormous amount of outliers in Insulin, Blood Pressure and BMI

# plt.subplots(figsize=(15, 10))
#
# plt.subplot(2,4,1)
# hist = plt.hist(diabetes["Pregnancies"])
#
# plt.subplot(2,4,2)
# hist = plt.hist(diabetes["Glucose"])
#
# plt.subplot(2,4,3)
# hist = plt.hist(diabetes["BloodPressure"])
#
# plt.subplot(2,4,4)
# hist = plt.hist(diabetes["SkinThickness"])
#
# plt.subplot(2,4,5)
# hist = plt.hist(diabetes["Insulin"])
#
# plt.subplot(2,4,6)
# hist = plt.hist(diabetes["BMI"])
#
# plt.subplot(2,4,7)
# hist = plt.hist(diabetes["DiabetesPedigreeFunction"])
#
# plt.subplot(2,4,8)
# hist = plt.hist(diabetes["Age"])
#
# plt.show()

# diabetes.hist(figsize=(8,8), xrot=45)
# plt.show()

# Multivariate Analysis

# Correlation
correlations = diabetes.corr()
# print(correlations)


plt.figure(figsize=(10,10))
sns.heatmap(correlations, cmap='RdBu_r', annot=True, vmax=1, vmin=-1)
plt.show()




