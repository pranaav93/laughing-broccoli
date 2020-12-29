#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pandas and numpy
import pandas as pd
import numpy as np

#Read Train data and test data
df = pd.read_csv(r'C:\Users\prana\OneDrive\Documents\Machine Learning\Multiple regression\House price prediction\train.csv')
df1 = pd.read_csv(r'C:\Users\prana\OneDrive\Documents\Machine Learning\Multiple regression\House price prediction\test.csv')

#drop columns with no sigificance or null values greater than 50%

df = df.drop(labels = ['Id', 'Fireplaces','FireplaceQu', 'PoolArea','PoolQC', 'Fence', 'MiscFeature','MiscVal', 'MoSold', 'YrSold'], axis =1)

#replace NA with 0 or none for non - categorical/categorical values

df["LotFrontage"].fillna(0, inplace = True)  
df["Alley"].fillna("No", inplace = True)  
df["MasVnrType"].fillna("None", inplace = True)  
df["MasVnrArea"].fillna(0, inplace = True)
df["BsmtQual"].fillna("No", inplace = True)
df["BsmtCond"].fillna("No", inplace = True)
df["BsmtExposure"].fillna("No", inplace = True)
df["BsmtFinType1"].fillna("No", inplace = True)
df["BsmtFinType2"].fillna("No", inplace = True)
df["GarageType"].fillna("No", inplace = True)
df["GarageYrBlt"].fillna(2020, inplace = True)
df["GarageFinish"].fillna("No", inplace = True)
df["GarageQual"].fillna("No", inplace = True)
df["GarageCond"].fillna("No", inplace = True)
df["Electrical"].fillna("No", inplace = True)
df["BuildingAge"] = ""

#calculate building age from date field and then remove the date fields after calculation
#assuming average life if there has been a remod done.

for i in range(len(df)):
    
    if(df['YearBuilt'][i] == df['YearRemodAdd'][i]) :
        
        df['BuildingAge'][i] = 2020 - (df['YearBuilt'][i])
        
    else :
        
        df['BuildingAge'][i] = (2020 - (df['YearBuilt'][i]) + (2020 - df['YearRemodAdd'][i]))
        df['BuildingAge'][i] = df['BuildingAge'][i] / 2
        
df["GarageAge"] = 2020 - df["GarageYrBlt"]
df = df.drop(labels = ["YearBuilt", "GarageYrBlt", "YearRemodAdd"], axis =1)

#Rearrange responsor column to last
df.columns
cols = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF',
       '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
       'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'GarageType',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'SaleType', 'SaleCondition',
        'BuildingAge', 'GarageAge', 'SalePrice']

df = df [cols]

#split x and Y variables.
x_train = df.iloc[:,:-1].values
y_train = df.iloc[:,69].values
x_train = pd.DataFrame(x_train)
x_train.columns = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF',
       '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
       'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'GarageType',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'SaleType', 'SaleCondition',
        'BuildingAge', 'GarageAge']

#perform preprocessing for x_test same as x_train:

df1.drop(labels = ['Id', 'Fireplaces','FireplaceQu', 'PoolArea','PoolQC', 'Fence', 'MiscFeature','MiscVal', 'MoSold', 'YrSold'], axis =1)

#replace NA with 0 or none for non - categorical/categorical values

df1["MSZoning"].fillna("No", inplace = True)  
df1["LotFrontage"].fillna(0, inplace = True)  
df1["Alley"].fillna("No", inplace = True)  
df1["Utilities"].fillna("No", inplace = True)
df1["Exterior1st"].fillna("No", inplace = True)
df1["Exterior2nd"].fillna("No", inplace = True)
df1["MasVnrType"].fillna("None", inplace = True)  
df1["MasVnrArea"].fillna(0, inplace = True)
df1["BsmtQual"].fillna("No", inplace = True)
df1["BsmtCond"].fillna("No", inplace = True)
df1["BsmtExposure"].fillna("No", inplace = True)
df1["BsmtFinType1"].fillna("No", inplace = True)
df1["BsmtFinSF1"].fillna(0, inplace = True)
df1["BsmtFinType2"].fillna("No", inplace = True)
df1["BsmtFinSF2"].fillna(0, inplace = True)
df1["BsmtUnfSF"].fillna(0, inplace = True)
df1["TotalBsmtSF"].fillna(0, inplace = True)
df1["BsmtFullBath"].fillna(0, inplace = True)
df1["BsmtHalfBath"].fillna(0, inplace = True)
df1["KitchenQual"].fillna(0, inplace = True)
df1["Functional"].fillna("None", inplace = True)
df1["GarageType"].fillna("No", inplace = True)
df1["GarageYrBlt"].fillna(2020, inplace = True)
df1["GarageFinish"].fillna("No", inplace = True)
df1["GarageArea"].fillna(0, inplace = True)
df1["GarageCars"].fillna(0, inplace = True)
df1["GarageQual"].fillna("No", inplace = True)
df1["GarageCond"].fillna("No", inplace = True)
df1["Electrical"].fillna("No", inplace = True)
df1["BuildingAge"] = ""

#calculate building age from date field and then remove the date fields after calculation
#assuming average life if there has been a remod done.

for i in range(len(df1)):
    
    if(df1['YearBuilt'][i] == df1['YearRemodAdd'][i]) :
        
        df1['BuildingAge'][i] = 2020 - (df1['YearBuilt'][i])
        
    else :
        
        df1['BuildingAge'][i] = (2020 - (df1['YearBuilt'][i]) + (2020 - df1['YearRemodAdd'][i]))
        df1['BuildingAge'][i] = df1['BuildingAge'][i] / 2
        
df1["GarageAge"] = 2020 - df1["GarageYrBlt"]
df1 = df1.drop(labels = ["YearBuilt", "GarageYrBlt", "YearRemodAdd"], axis =1)

#Rearrange responsor column to last
df1.columns
cols = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF',
       '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
       'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'GarageType',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'SaleType', 'SaleCondition',
        'BuildingAge', 'GarageAge']

x_test = df1 [cols]

# concat x_train and x_test for preprocessing :

x_values = pd.DataFrame()
x_values = pd.concat([x_train,x_test])


# In[2]:


#creating dummies / 1 hot encoding for categorical vriables:

x_dummies = pd.get_dummies(data = x_values, columns= ['MSSubClass',
                                               'MSZoning',
                                               'Street',
                                               'Alley',
                                               'LotShape', 
                                               'LandContour', 
                                               'Utilities',
                                              'LotConfig',
                                              'LandSlope',
                                              'Neighborhood',
                                              'Condition1',
                                              'Condition2',
                                              'BldgType',
                                              'HouseStyle',
                                              'OverallQual',
                                              'OverallCond',
                                              'RoofStyle',
                                              'RoofMatl',
                                              'Exterior1st',
                                              'Exterior2nd',
                                              'MasVnrType',
                                              'ExterQual',
                                              'ExterCond',
                                              'Foundation',
                                              'BsmtQual',
                                              'BsmtCond',
                                              'BsmtExposure',
                                               'BsmtFinType1',
                                               'BsmtFinType2',
                                               'Heating',
                                               'HeatingQC',
                                               'CentralAir',
                                               'Electrical',
                                               'KitchenQual',
                                               'Functional',
                                               'GarageType',
                                               'GarageFinish',
                                               'GarageQual',
                                               'GarageCond',
                                               'PavedDrive',
                                               'SaleType',
                                               'SaleCondition'])

x_dummies['intercept']=1

# Convert numerical variables from object to float (pandas defaultly convert every column to object and VIF cannot be run on dtype = objects):
x_dummies['LotFrontage']=x_dummies['LotFrontage'].astype('float64')
x_dummies['LotArea']=x_dummies['LotArea'].astype('float64')
x_dummies['MasVnrArea']=x_dummies['MasVnrArea'].astype('float64')
x_dummies['BsmtFinSF1']=x_dummies['BsmtFinSF1'].astype('float64')
x_dummies['BsmtFinSF2']=x_dummies['BsmtFinSF2'].astype('float64')
x_dummies['BsmtUnfSF']=x_dummies['BsmtUnfSF'].astype('float64')
x_dummies['TotalBsmtSF']=x_dummies['TotalBsmtSF'].astype('float64')
x_dummies['1stFlrSF']=x_dummies['1stFlrSF'].astype('float64')
x_dummies['2ndFlrSF']=x_dummies['2ndFlrSF'].astype('float64')
x_dummies['LowQualFinSF']=x_dummies['LowQualFinSF'].astype('float64')
x_dummies['GrLivArea']=x_dummies['GrLivArea'].astype('float64')
x_dummies['BsmtFullBath']=x_dummies['BsmtFullBath'].astype('float64')
x_dummies['BsmtHalfBath']=x_dummies['BsmtHalfBath'].astype('float64')
x_dummies['FullBath']=x_dummies['FullBath'].astype('float64')
x_dummies['HalfBath']=x_dummies['HalfBath'].astype('float64')
x_dummies['KitchenAbvGr']=x_dummies['KitchenAbvGr'].astype('float64')
x_dummies['BedroomAbvGr']=x_dummies['BedroomAbvGr'].astype('float64')
x_dummies['TotRmsAbvGrd']=x_dummies['TotRmsAbvGrd'].astype('float64')
x_dummies['GarageCars']=x_dummies['GarageCars'].astype('float64')
x_dummies['GarageArea']=x_dummies['GarageArea'].astype('float64')
x_dummies['WoodDeckSF']=x_dummies['WoodDeckSF'].astype('float64')
x_dummies['OpenPorchSF']=x_dummies['OpenPorchSF'].astype('float64')
x_dummies['EnclosedPorch']=x_dummies['EnclosedPorch'].astype('float64')
x_dummies['3SsnPorch']=x_dummies['3SsnPorch'].astype('float64')
x_dummies['ScreenPorch']=x_dummies['ScreenPorch'].astype('float64')
x_dummies['BuildingAge']=x_dummies['BuildingAge'].astype('float64')
x_dummies['GarageAge']=x_dummies['GarageAge'].astype('float64')


# In[3]:


# function to calculate VIF systematiclly and remove columns that has vif > 3, returns the final dataframe with clean columns

from statsmodels.stats.outliers_influence import variance_inflation_factor  
import numpy as np

def calculate_vif_(X, thresh=100):
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables = np.delete(variables, maxloc)
            dropped=True

    print('Remaining variables:')
    print(X.columns[variables])
    return X[cols[variables]]


# In[4]:


# Invoke calculate_vif function and store it in new dataframe
x_updated = calculate_vif_(x_dummies, thresh = 3.0)


# In[7]:


#splitting data frame into train and test records
x_train = x_updated.iloc[:1460,:]
x_test = x_updated.iloc[1460:,:]


# In[8]:


#splitting test and train records internally from x_train data set to create validation set
from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1 = train_test_split(x_train,y_train,test_size=0.25)


# In[9]:


# develop regression model and predict for x_test1

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train1,y_train1)
y_pred = regression.predict(x_test1)
print(y_pred)


# In[12]:


# calculate r2 and adjusted r2 values

from sklearn.metrics import r2_score
r2 = r2_score(y_test1,y_pred)
adj_r2 = 1-((1-r2)*(365-1)/(365-189-1))
print (r2)
print (adj_r2)


# In[20]:


# predict values for x_test and export it to excel

y_pred_test = regression.predict(x_test)
print(y_pred_test)

# creating a list of index names for y_pred_test
#index_values = ['first'] 
   
# creating a list of column names for y_pred_test
column_values = ['SalePrice'] 
y_pred_test = pd.DataFrame(data = y_pred_test, 
                  columns = column_values)  

# export dataframe to excel

y_pred_test.to_excel(r'C:\Users\prana\OneDrive\Documents\y_pred_output1.xlsx', index = False)


# In[ ]:




