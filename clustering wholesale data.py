#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r'C:\Users\aios210720\Downloads\cluster Example.csv')


# In[3]:


df.columns


# In[4]:


sns.scatterplot(x="Satisfaction",y="Loyalty",data=df)


# In[6]:


from sklearn.cluster import KMeans


# In[18]:


sum_squ_diss=[]
for i in range(1,10):
    km=KMeans(n_clusters=i)
    km=km.fit(df)
    sum_squ_diss.append(km.inertia_)


# In[8]:


sum_squ_diss


# In[9]:


k=range(1,10)
plt.plot(k,sum_squ_diss)


# In[10]:


algo=KMeans(n_clusters=4,max_iter=1000)
algo.fit(df)


# In[11]:


cen=algo.cluster_centers_


# In[12]:


cen


# In[13]:


plt.scatter(df["Satisfaction"],df["Loyalty"],c=algo.labels_)
plt.scatter(cen[:,0],cen[:,1],c='red')


# In[14]:


df["predict"]=algo.fit_predict(df)


# In[15]:


df


# In[ ]:




