# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 18:41:54 2021

@author: utkar
"""

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# part a
titanic_utkarsh = pd.read_csv('titanic.csv')

#part b
# # displaying the first 3 records
# print('DATAFRAME HEAD SAMPLE\n', titanic_utkarsh.head(3))

# # displaying the sahpe of the df
# print('\nDATAFRAME SHAPE\n', titanic_utkarsh.shape)

# # displaying the info
# print('\nDATAFRAME INFO\n')
# print(titanic_utkarsh.info(show_counts=True))


# # print('>> Example unique information id : ', titanic_utkarsh['PassengerId'][30])
# # print('>> Example passenger name : ', titanic_utkarsh['Name'][30])
# # print('>> Example ticket number : ', titanic_utkarsh['Ticket'][30])

# print('\nnon null values in  Passenger Id column : ', titanic_utkarsh['PassengerId'].notna().sum())
# #print(titanic_utkarsh['PassengerId'].isna().sum())
# print('Unique values in  Passenger Id column : ', len(titanic_utkarsh['PassengerId'].unique()))

# print('\nnon null values in  Name column : ',titanic_utkarsh['Name'].notna().sum())
# #print(titanic_utkarsh['Name'].isna().sum())
# print('Unique values in Name column : ',len(titanic_utkarsh['Name'].unique()))

# print('\nnon null values in  Ticket column : ',titanic_utkarsh['Ticket'].notna().sum())
# #print(titanic_utkarsh['Ticket'].isna().sum())
# print('Unique values in Ticket column : ', len(titanic_utkarsh['Ticket'].unique()))

# print('\nnon null values in Cabin column : ',titanic_utkarsh['Cabin'].notna().sum())
# #print(titanic_utkarsh['Cabin'].isna().sum())
# print('Unique values in Cabin column : ', len(titanic_utkarsh['Cabin'].unique()))

# print('\nUnique values in Sex column : ',titanic_utkarsh['Sex'].unique())
# print('\nUnique values in Pclass column : ',titanic_utkarsh['Pclass'].unique())

#part c
"""
    DATA VISUALIZATION 
"""
# print('\n\nBAR CHARTS FOR SURVIVALS COMPARISON')
# pd.crosstab(titanic_utkarsh["Pclass"], titanic_utkarsh["Survived"]).plot(kind='bar')
# plt.legend(['Yes','No'])
# plt.title('Survived by Class (utkarsh)')
# plt.xlabel('Class')
# plt.ylabel('Frequency (Survived)')

# pd.crosstab(titanic_utkarsh.Sex, titanic_utkarsh.Survived,).plot(kind='bar')
# plt.legend(['Yes','No'])
# plt.title('Survived by Gender (utkarsh)')
# plt.xlabel('Gender')
# plt.ylabel('Frequency (Survived)')

# print('\n\nSCATTER MATRIX PLOT')
# pd.plotting.scatter_matrix(titanic_utkarsh[['Survived', 'Sex', 'Pclass', 'Fare', 'SibSp', 'Parch']], alpha=0.2, figsize=(13, 15))
# plt.savefig('./relationship_scattermatrix.png')


# titanic_utkarsh['Survived'].hist()
# plt.title('Survived Histogram')

# part d

# """
#     DATA TRANSFORMATION
# """

# print('\n\nTRANSFORMING COLUMNS')
titanic_utkarsh_set = titanic_utkarsh.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# print('DATAFRAME HEAD SAMPLE\n', titanic_utkarsh_set.head(3))
# print('\nDATAFRAME SHAPE\n', titanic_utkarsh.shape)


# useless
# categorical_vars = ['Sex', 'Embarked']
# for var in categorical_vars:
#     categorical_var_dummy = pd.get_dummies(titanic_utkarsh_set[var], prefix=var)
#     titanic_utkarsh_set = titanic_utkarsh_set.join(categorical_var_dummy)

# Transforming categorical values into numerical values
titanic_utkarsh_set = pd.get_dummies(titanic_utkarsh_set,columns=['Sex', 'Embarked'])
# print('DATAFRAME HEAD SAMPLE\n', titanic_utkarsh_set.head(3))
# print('\nDATAFRAME SHAPE\n', titanic_utkarsh_set.dtypes)



# titanic_utkarsh_set = titanic_utkarsh_set.drop(categorical_vars, axis=1)

titanic_utkarsh_set['Age'].fillna(
    value=titanic_utkarsh_set['Age'].mean(), inplace=True)
titanic_utkarsh_set = titanic_utkarsh_set.astype('float64')
# print(titanic_utkarsh_set.info(show_counts=True))

# print('\n null values in  Age column : ',titanic_utkarsh_set['Age'].isna().sum())



# def normalize_dataframe(dataframe):
#     """
#     This function normalizes the values for a dataframe with all numeric 
#     columns

#     Parameters
#     ----------
#     dataframe 
#         All numeric dataframe

#     Returns
#     -------
#     Normalized dataframe

#     """
#     for col in dataframe.columns.values:
#         min_col = dataframe[col].min()
#         max_col = dataframe[col].max()
#         dataframe[col] = dataframe[col].apply(
#             lambda x: ((x-min_col)/(max_col-min_col)))

#     return dataframe

# titanic_utkarsh_normal = normalize_dataframe(titanic_utkarsh_set)

# print('DATAFRAME HEAD SAMPLE\n', titanic_utkarsh_normal.head(3))


# Function for normalizing all data points using min and max values
def normalized_data(df):
    
      df = (df - df.min()) / (df.max() -df.min())
      print("\n--->normalized dataframe\n")
      return df

titanic_utkarsh_normal = normalized_data(titanic_utkarsh_set)
print('DATAFRAME HEAD SAMPLE\n', titanic_utkarsh_normal.head(2))




titanic_utkarsh_normal.hist(figsize=(9, 10))

titanic_utkarsh_normal[['Embarked_C', 'Embarked_Q',
                        'Embarked_S']].hist(figsize=(9, 10))

y_utkarsh = titanic_utkarsh_normal['Survived']
x_utkarsh = titanic_utkarsh_normal.drop('Survived', axis=1)

# Last two digits of student id as seed
x_train_utkarsh, x_test_utkarsh, y_train_utkarsh, y_test_utkarsh = train_test_split(
    x_utkarsh, y_utkarsh, test_size=0.3, random_state=79)

# """
#     LOGISTIC REGRESSION MODEL
# """

# # Fit Logistic Regression Model
# utkarsh_model = linear_model.LogisticRegression(solver='lbfgs')
# utkarsh_model.fit(x_train_utkarsh, y_train_utkarsh)

# print('\n\nDISPLAY MODEL COEFFICIENTS')
# coef_df = pd.DataFrame(
#     zip(x_train_utkarsh.columns, np.transpose(utkarsh_model.coef_)))
# print(coef_df)


# print('\n\nDISPLAY CROSS-VALIDATION RESULTS')
# print('Test Size - Min Mean Max Range')
# for ts in np.arange(0.10, 0.55, 0.05):
#     x_train_cross, x_test_cross, y_train_cross, y_test_cross = train_test_split(
#         x_utkarsh, y_utkarsh, test_size=ts, random_state=31)

#     scores_cross = cross_val_score(linear_model.LogisticRegression(
#         solver='lbfgs'), x_train_cross, y_train_cross, scoring='accuracy', cv=10)

#     score_min = scores_cross.min()
#     score_max = scores_cross.max()
#     score_mean = scores_cross.mean()
#     line = "Test Size: {0:.4f}  || Metrics: {1:.4f}   {2:.4f}   {3:.4f}   {4:.4f}".format(
#         ts, score_min, score_mean, score_max, score_max - score_min)
#     print(line)


# """
#     MODEL TESTING
# """
# # Last two digits of student id as seed
# x_train_utkarsh, x_test_utkarsh, y_train_utkarsh, y_test_utkarsh = train_test_split(
#     x_utkarsh, y_utkarsh, test_size=0.3, random_state=31)
# # Fit Logistic Regression Model
# utkarsh_model = linear_model.LogisticRegression(solver='lbfgs')
# utkarsh_model.fit(x_train_utkarsh, y_train_utkarsh)

# print("\n\n*** METRICS 0.5 THRESHOLD ***")

# y_pred_utkarsh = utkarsh_model.predict_proba(x_test_utkarsh)
# y_pred_utkarsh_flag = y_pred_utkarsh[:, 1] > 0.5
# y_predicted = y_pred_utkarsh_flag.astype(int)
# y_predicted = np.array(y_predicted)

# cmatrix = confusion_matrix(y_test_utkarsh.values, y_predicted)
# ascore = accuracy_score(y_test_utkarsh.values, y_predicted)
# creport = classification_report(y_test_utkarsh.values, y_predicted)


# print("\n>> CONFUSION MATRIX\n", cmatrix)
# print("\n>> ACCURACY SCORE\n", ascore)
# print("\n>> CLASSIFICATION REPORT\n", creport)


# print("\n\n*** METRICS 0.75 THRESHOLD ***")

# # y_pred_utkarsh = utkarsh_model.predict_proba(x_test_utkarsh)
# y_pred_utkarsh_flag = y_pred_utkarsh[:, 1] > 0.75
# y_predicted2 = y_pred_utkarsh_flag.astype(int)
# y_predicted2 = np.array(y_predicted2)

# cmatrix2 = confusion_matrix(y_test_utkarsh.values, y_predicted2)

# ascore2 = accuracy_score(y_test_utkarsh.values, y_predicted2)
# creport2 = classification_report(y_test_utkarsh.values, y_predicted2)

# print("\n>> CONFUSION MATRIX\n", cmatrix2)
# print("\n>> ACCURACY SCORE\n", ascore2)
# print("\n>> CLASSIFICATION REPORT\n", creport2)

# tn, fp, fn, tp = confusion_matrix(y_test_utkarsh.values, y_predicted2).ravel()