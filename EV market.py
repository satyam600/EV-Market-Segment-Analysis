#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


# In[35]:


df=pd.read_csv('electric_vehicle_charging_station_list.csv')
df


# In[4]:


sns.heatmap (df.isnull(), yticklabels=False, annot=True)


# In[36]:


df.isnull().sum()


# In[37]:


df=df.drop('no', axis='columns')
df.head(3)


# In[16]:


plt.figure(figsize=(15,12))
sns.countplot(x = 'region', data = df)


# In[18]:


plt.figure(figsize=(10,8))
sns.countplot(x = 'power', data = df)


# In[20]:


sns.relplot(x="latitude", y="longitude", height=6,hue="region",data=df)


# In[13]:


fig = px.density_mapbox(df, lat='latitude', lon='longitude', radius=1, center=dict(lat=21, lon=79), zoom=1.5, mapbox_style="stamen-terrain")
fig.show()


# In[58]:


X1=df.drop(['region','address','aux addres','type','power','service'], axis='columns')
X1


# In[59]:


y1=df.region
y1


# In[71]:


X_train, X_test, y_train, y_test=train_test_split(X1,y1, test_size=0.2)


# In[72]:


model1=SVC()


# In[73]:


model1.fit(X_train, y_train)


# In[74]:


model1.score(X_test, y_test)


# In[ ]:




