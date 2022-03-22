#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[9]:


df=pd.read_csv(r'C:\Users\aios210720\Downloads\international-airline-passengers.csv',skipfooter=2)


# In[10]:


df.shape


# In[11]:


df.head()


# In[12]:


df.tail()


# In[13]:


df.info()


# In[14]:


df["Month"]=pd.to_datetime(df["Month"])


# In[16]:


df.info()


# In[17]:


df.head()


# In[18]:


df1=df.drop("Month",axis=1)
df1.columns=["volume"]
df1.index=df["Month"]
df1.head()


# In[20]:


plt.plot(df1)
plt.show()


# In[21]:


rolling_mean=df1.rolling(10).mean()
rolling_std=df1.rolling(10).std()
#df1.rolling(10)
plt.plot(rolling_mean,c='r')
plt.plot(rolling_std,c='g')
plt.show()


# # dicky fuller test

# In[22]:


from statsmodels.tsa.stattools import adfuller
adfuller(df1["volume"])


# In[23]:


dflog=np.log(df1)
plt.plot(dflog)
plt.show()


# In[24]:


dflog=dflog-dflog.shift(1)
dflog.head()


# In[25]:


rolling_mean1=dflog.rolling(10).mean()
rolling_std1=dflog.rolling(10).std()
#df1.rolling(10)
plt.plot(rolling_mean1,c='r')
plt.plot(rolling_std1,c='g')
plt.show()


# In[26]:


from statsmodels.tsa.stattools import adfuller
adfuller(dflog["volume"].dropna())


# In[ ]:




