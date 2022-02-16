#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
data = pd.read_csv("car data.csv")
data.head()


# In[4]:


data['Years_old'] = 2022-data['Year']


# In[5]:


Final_data = data[['Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner','Years_old']]


# In[6]:


Final_data = pd.get_dummies(Final_data,drop_first = True)


# In[7]:


#dependent and independent feature
x = Final_data.iloc[:,1:]
y = Final_data.iloc[:,0]


# In[8]:


#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


# In[9]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train.values,y_train.values)


# In[10]:


prediction = lr.predict(x_test.values)


# In[11]:


lr.score(x_test.values,y_test.values)


# In[12]:


import pickle


# In[13]:


file = open('model.pkl','wb')
pickle.dump(lr,file)


# In[ ]:




