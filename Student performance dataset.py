#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd


# In[3]:


df=pd.read_csv(r'C:\Users\aios210720\Downloads\data.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[8]:


df.columns


# In[9]:


df.dtypes


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df_dup=df.drop_duplicates()

df.shape


# In[13]:


plt.figure(figsize=(20,10))
plt.title("Heatmap of continuous features",fontweight='bold',fontsize=20)
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu',linewidth=1)


# In[14]:


df['math score'].median(),df['writing score'].median(),df['reading score'].median()


# In[15]:


df['overall_score']=df['math score']+df['reading score']+df['writing score']
df.head()


# In[16]:


sns.jointplot(data=df,x='writing score',y='math score',palette='rocket',hue='gender')


# In[17]:


sns.jointplot(data=df,x='writing score',y='math score',palette='rocket',hue='lunch')


# In[18]:


sns.jointplot(data=df,x='reading score',y='math score',palette='rocket',hue='gender')


# In[19]:


sns.jointplot(data=df,x='reading score',y='math score',palette='rocket',hue='lunch')


# In[20]:


sns.jointplot(data=df,x='reading score',y='writing score',palette='rocket',hue='gender')


# In[21]:


sns.jointplot(data=df,x='reading score',y='writing score',palette='rocket',hue='lunch')


# In[22]:


df=df.drop(['math score','writing score','reading score'],axis=1)
df.head()


# In[23]:


df['gender']=df['gender'].map({'female':0 , 'male':1}).astype(int)
df['lunch']=df['lunch'].map({'standard':1 , 'free/reduced':0}).astype(int)
df['test preparation course']=df['test preparation course'].map({'none':0 , 'completed':1}).astype(int)
df


# In[24]:


df=pd.get_dummies(df)
df


# In[25]:


Y=df['overall_score']
X=df.drop('overall_score',axis=1)
X.head()


# In[26]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error


# In[27]:


#SPLITTING
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


# In[28]:


#MODEL
model=RandomForestRegressor()
model.fit(x_train,y_train)


# In[29]:


y_pred = model.predict(x_test)


# In[30]:


print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))


# In[31]:


feature_importance = np.array(model.feature_importances_)
feature_names = np.array(x_train.columns)
data={'feature_names':feature_names,'feature_importance':feature_importance}
df_plt = pd.DataFrame(data)
df_plt.sort_values(by=['feature_importance'], ascending=False,inplace=True)
plt.figure(figsize=(10,8))
sns.barplot(x=df_plt['feature_importance'], y=df_plt['feature_names'])
plt.xlabel('FEATURE IMPORTANCE')
plt.ylabel('FEATURE NAMES')
plt.show()


# In[ ]:




