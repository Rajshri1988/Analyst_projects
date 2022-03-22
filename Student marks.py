#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd


# In[3]:


df=pd.read_csv(r'C:\Users\aios210720\Downloads\data (1).csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.dtypes


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df_dup=df.drop_duplicates()

df.shape


# In[11]:


plt.figure(figsize=(20,10))
plt.title("Heatmap of continuous features",fontweight='bold',fontsize=20)
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu',linewidth=1)


# In[12]:


df['math score'].median(),df['writing score'].median(),df['reading score'].median()


# In[13]:


df['overall_score']=df['math score']+df['reading score']+df['writing score']
df.head()


# In[14]:


sns.jointplot(data=df,x='writing score',y='math score',palette='rocket',hue='gender')


# In[15]:


sns.jointplot(data=df,x='writing score',y='math score',palette='rocket',hue='lunch')


# In[16]:


sns.jointplot(data=df,x='reading score',y='math score',palette='rocket',hue='gender')


# In[17]:


sns.jointplot(data=df,x='reading score',y='math score',palette='rocket',hue='lunch')


# In[18]:


sns.jointplot(data=df,x='reading score',y='writing score',palette='rocket',hue='gender')


# In[19]:


sns.jointplot(data=df,x='reading score',y='writing score',palette='rocket',hue='lunch')


# In[20]:


df=df.drop(['math score','writing score','reading score'],axis=1)
df.head()


# In[21]:


df['gender']=df['gender'].map({'female':0 , 'male':1}).astype(int)
df['lunch']=df['lunch'].map({'standard':1 , 'free/reduced':0}).astype(int)
df['test preparation course']=df['test preparation course'].map({'none':0 , 'completed':1}).astype(int)
df


# In[22]:


df=pd.get_dummies(df)
df


# In[23]:


Y=df['overall_score']
X=df.drop('overall_score',axis=1)
X.head()


# In[ ]:




