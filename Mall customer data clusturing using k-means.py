#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


df=pd.read_csv(r'C:\Users\aios210720\Downloads\shopping-data.csv')


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df=df.drop_duplicates()


# In[10]:


df.isnull().sum()


# In[12]:


df.dtypes


# In[13]:


a=["Age","Annual Income (k$)","Spending Score (1-100)"]


# In[14]:


for i in a:
    print(df[i].describe())
    print(df[i].skew())
    sns.distplot(df[i])
    plt.show()


# In[15]:


sns.countplot(x="Genre",data=df)


# In[17]:


df.iloc[:,3:5]


# In[18]:


df.head()


# In[19]:


from sklearn.cluster import KMeans
algo=KMeans(n_clusters=2)
algo.fit(data)


# In[20]:


cen=algo.cluster_centers_


# In[21]:


algo.labels_


# In[22]:


algo.inertia_


# In[23]:


sns.scatterplot(data['Annual Income (k$)'],data['Spending Score (1-100)'],hue=algo.labels_)
sns.scatterplot(cen[:,0],cen[:,1],color='r')


# In[24]:


dis=[]
k=range(1,15)
for i in k:
    algo=KMeans(n_clusters=i)
    algo.fit(data)
    dis.append(algo.inertia_)


# In[25]:


dis


# In[26]:


plt.plot(k,dis)
plt.show()


# In[27]:


algo1=KMeans(n_clusters=5)
algo1.fit(data)


# In[28]:


KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=5, n_init=10, n_jobs=None, precompute_distances='auto',
       random_state=None, tol=0.0001, verbose=0)


# In[29]:


cen=algo1.cluster_centers_
cen


# In[30]:


sns.scatterplot(data['Annual Income (k$)'],data['Spending Score (1-100)'],hue=algo1.labels_)
sns.scatterplot(cen[:,0],cen[:,1],color='r')


# In[ ]:




