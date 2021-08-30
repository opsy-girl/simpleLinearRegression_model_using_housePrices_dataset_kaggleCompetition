#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


#Import the data file
HouseTrain = pd.read_csv(r'C:\Users\XXX\Downloads\home-data-for-ml-course\HomeTrain.csv')


# In[5]:


#Quick view of data summary information
HouseTrain.info()
#HouseTrain[HouseTrain.isnull()==True]


# In[6]:


#Confirm the count of null values in a given column
HouseTrain['Alley'].isna().count()


# In[8]:


#Check the significance of all the independent variables against the dependent variables
HouseTrain.corr()['SalePrice'].sort_values()


# In[29]:


#Perform a simple visualization of one of the independent variables against the dependent variable "Saleprice"
plt.scatter(HouseTrain['SalePrice'], HouseTrain['TotalBsmtSF'])
plt.show()

# The significance shown below is of a strong positive correlation


# In[7]:


#Assign a dependent variable in preparation for the machine learning model
y = HouseTrain['SalePrice']


# In[30]:


#Assign the independent variables in preparation for the machine learning model
x = HouseTrain[['OverallQual','GrLivArea','GarageCars', 'TotalBsmtSF','1stFlrSF','KitchenAbvGr', 'EnclosedPorch' ]]


# In[11]:


#Import the model
from sklearn.model_selection import train_test_split


# In[12]:


#Split the train data into 70,30
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[13]:


#Import the linear regression model for this data insight. Note, linear regression is used for numeric insights generation
from sklearn.linear_model import LinearRegression


# In[14]:


#Initialize the model
model = LinearRegression()


# In[15]:


#Fit the model
model.fit(x_train, y_train)


# In[16]:


#Get the model co-efficient values
print(model.coef_)


# In[17]:


#Get the model intercept
print(model.intercept_)


# In[18]:


#Print the model co-efficient values in a table
pd.DataFrame(model.coef_, x.columns, columns = ['Coeff'])


# In[21]:


#Here is a the mathematical model formed by the linear regression function for evaluation test values. 

y= -74724 + 23909[OverallQual] + 45[GrLivArea] + 17504[GarageCars]+ 16[TotalBsmtSF]+18[1stFlrSF]-26607[KitchenAbvGr]-46[EnclosedPorch]
print('-74724 + 23909[OverallQual] + 45[GrLivArea] + 17504[GarageCars]+ 16[TotalBsmtSF]+18[1stFlrSF]-26607[KitchenAbvGr]-46[EnclosedPorch]')


# In[122]:


#Fit the model to the data

#LogisticRegression(random_state=0).fit(x_train, y_train)
model.fit(x_train, y_train)


# In[23]:


# Returns a NumPy Array
# Predict for test data
predictions = model.predict(x_test)


# In[24]:


#Chect the model accuracy
score = model.score(x_test, y_test)
print(score)


# In[ ]:




