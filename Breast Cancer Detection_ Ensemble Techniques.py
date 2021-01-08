#!/usr/bin/env python
# coding: utf-8

# In[32]:


#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[33]:


# read Data 
df= pd.read_csv(r"C:\Users\prana\OneDrive\Documents\Machine Learning\Classification\BreastCancer\data.csv")


# In[34]:


# Exploratrory Data analysis

df.info() # no null values
df.shape # 33 features present


# In[38]:


df = df.drop(labels = ['id', 'Unnamed: 32'], axis = 1) #drop unwanted columns id , Unnamed: 32
df.describe()

df.head()


# In[9]:


# check for duplicates

duplicate = df.duplicated()
print (duplicate.sum())

df[duplicate]

# no duplicates found


# In[45]:


# check and Treat outliers if present for columns

cols = list(df.columns)

cols = cols.remove('diagnosis')

plt.figure(figsize = (32,32))

df.boxplot(column = cols)


# In[50]:


# columns perimeter_mean, area_mean, perimeter_se, area_se, perimeter_worst, area_worst has outliers.

# create a function to find the lower range and upper range from IQR.

def get_IQRRange(col):
    
    sorted(col) # sort column values in ascending order
    Q1, Q3 = col.quantile([0.25, 0.75]) # quarter 1 and quarter 3 values
    IQR = Q3 - Q1
    upper_range = Q3 + (1.5 * IQR)
    lower_range = Q1 - (1.5 * IQR)
    
    return lower_range, upper_range


# In[52]:


# remove outliers and replace with upper and lower ranges for the 6 columns.

#Finding lower and upper range from IQR

low_periMean, up_periMean = get_IQRRange(df['perimeter_mean'])
low_areaMean, up_areaMean = get_IQRRange(df['area_mean'])
low_periSE, up_periSE = get_IQRRange(df['perimeter_se'])
low_areaSE, up_areaSE = get_IQRRange(df['area_se'])
low_periWors, up_periWors = get_IQRRange(df['perimeter_worst'])
low_areaWors, up_areaWors = get_IQRRange(df['area_worst'])


# In[54]:


# replacing the outliers with lower and upper range for the columns

df['perimeter_mean'] = np.where (df['perimeter_mean'] > up_periMean, up_periMean, df['perimeter_mean'])
df['perimeter_mean'] = np.where (df['perimeter_mean'] < low_periMean, low_periMean, df['perimeter_mean'])
df['area_mean'] = np.where (df['area_mean'] > up_areaMean, up_areaMean, df['area_mean'])
df['area_mean']= np.where (df['area_mean'] < low_areaMean, low_areaMean, df['area_mean'])
df['perimeter_se'] = np.where (df['perimeter_se'] > up_periSE, up_periSE, df['perimeter_se'])
df['perimeter_se'] = np.where (df['perimeter_se'] < low_periSE, low_periSE, df['perimeter_se'])
df['area_se'] = np.where (df['area_se'] > up_areaSE, up_areaSE, df['area_se'])
df['area_se'] = np.where (df['area_se'] < low_areaSE, low_areaSE, df['area_se'])
df['perimeter_worst'] =  np.where (df['perimeter_worst'] > up_periWors, up_periWors, df['perimeter_worst'])
df['perimeter_worst'] = np.where (df['perimeter_worst'] < low_periWors, low_periWors, df['perimeter_worst'])
df['area_worst'] = np.where (df['area_worst'] > up_areaWors, up_areaWors, df['area_worst'])
df['area_worst'] = np.where (df['area_worst'] < low_areaWors, low_areaWors, df['area_worst'])


# In[56]:


# box plot after treating outliers:

plt.figure(figsize = (32,32))

df.boxplot(column = cols)


# In[76]:


# split predictor and responser variables.

y = df.iloc[:,0]
x = df.iloc[:,1:]


# In[78]:


# split test and train records

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 100)


# In[80]:


# scale values using standard scaler.

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit_transform(x_train)

sc.fit(x_test)


# ### Model Building Using Logistic Regression

# In[87]:


# building a logistic regression model and calculating accuracy scores.

from sklearn.linear_model import LogisticRegression

log_class = LogisticRegression()

log_class.fit(x_train, y_train)

y_pred = log_class.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix (y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# ## Model Building using Naive Bayes Classfier

# In[88]:


# building a Gaussian NB classifier model and calculating accuracy scores.

from sklearn.naive_bayes import GaussianNB

NB_class = GaussianNB()

NB_class.fit(x_train, y_train)

y_pred = NB_class.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix (y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# ## Model Building using K-NN

# In[89]:


# building a KNN classifier model and calculating accuracy scores.

from sklearn.neighbors import KNeighborsClassifier

KNN_class = KNeighborsClassifier(n_neighbors= 5, p=2,metric='minkowski')

KNN_class.fit(x_train, y_train)

y_pred = KNN_class.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix (y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# ## Model Building using Decision Trees

# In[91]:


# building a Decision Tree classifier model and calculating accuracy scores.

from sklearn.tree import DecisionTreeClassifier

DT_class = DecisionTreeClassifier(criterion = 'entropy')

DT_class.fit(x_train, y_train)

y_pred = DT_class.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix (y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# ## Model Building using Random Forests bagging technique.

# In[98]:


# building a Decision Tree classifier model and calculating accuracy scores.

from sklearn.ensemble import RandomForestClassifier

RF_class = RandomForestClassifier(n_estimators = 40, criterion = 'entropy')

RF_class.fit(x_train, y_train)

y_pred = RF_class.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix (y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# ### We can conclude that random Forest with default parameters explains the dataset with 95 % accuracy and with minimum number of False Positives 2 compared to others.

# In[ ]:




