#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv(r'C:\Users\aios210720\Downloads\maintenance_data (1).csv')


# In[4]:


df.shape


# In[5]:


df.info()


# In[7]:


df.columns


# In[12]:


df.dtypes


# In[14]:


df.head()


# In[15]:


con=["lifetime","pressureInd","moistureInd","temperatureInd"]
cat=["broken","team","provider"]


# In[16]:


df.duplicated().sum()


# In[17]:


df=df.drop_duplicates()


# In[18]:


df.shape


# In[19]:


df.isnull().sum()


# In[20]:


for i in cat:
    sns.countplot(x=i,data=df)
    plt.show()


# In[21]:


for i in con:
    sns.distplot(df[i])
    plt.show()


# In[22]:


for i in con:
    sns.scatterplot(x=df.index, y=df[i])
    plt.show()


# In[23]:


#bivariate analysis
for i in con:
    sns.swarmplot(x="broken",y=i,data=df)
    plt.show()


# In[24]:


for i in con:
    sns.scatterplot(df[i],df["lifetime"])
    plt.show()


# In[25]:


sns.heatmap(df.corr(),annot=True,cmap="coolwarm")


# In[26]:


for i in cat:
    sns.countplot(x=df[i],hue=df["broken"])
    plt.show()


# In[27]:


out=pd.crosstab(df["team"],df["broken"],margins=True)


# In[28]:


out


# In[29]:


out[1]/out["All"]


# In[30]:


out=pd.crosstab(df["provider"],df["broken"],margins=True)
out[1]/out["All"]


# In[31]:


#multicariate analysis
sns.scatterplot(x="lifetime",y="moistureInd",hue="broken",data=df)


# In[32]:


sns.scatterplot(x="lifetime",y="pressureInd",hue="broken",data=df)


# In[33]:


sns.scatterplot(x="lifetime",y="temperatureInd",hue="broken",data=df)


# In[34]:


sns.swarmplot(x="provider",y="moistureInd",hue="broken",data=df)


# In[35]:


sns.swarmplot(x="provider",y="temperatureInd",hue="broken",data=df)


# In[36]:


sns.swarmplot(x="provider",y="pressureInd",hue="broken",data=df)


# In[ ]:




