#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[5]:


df=pd.read_csv(r'C:\Users\aios210720\Downloads\insurance.csv')


# In[8]:


df


# In[9]:


df.shape


# In[11]:


df.columns


# In[12]:


df.isnull().sum()


# In[13]:


df.dtypes


# In[14]:


a=["sex","smoker","region"]
b=['age','bmi','children','charges']


# In[15]:


for i in a:
    print(df[i].value_counts())
    sns.countplot(x=i,data=df)
    plt.show()


# In[16]:


for i in b:
    sns.distplot(df[i],kde=False)
    print(df[i].describe())
    print(df[i].skew())
    plt.show()


# In[17]:


for i in a:
    sns.swarmplot(x=i,y="charges",data=df)
    plt.show()
    sns.boxplot(x=i,y="charges",data=df)
    plt.show()


# In[18]:


for i in b:
    sns.scatterplot(x=i,y="charges",data=df)
    print(df[[i,"charges"]].corr())
    plt.show()


# In[19]:


sns.scatterplot(x="age",y="charges",hue="smoker",data=df)
plt.show()


# In[20]:


sns.scatterplot(x="age",y="charges",hue="sex",data=df)
plt.show()


# In[21]:


sns.scatterplot(x="age",y="charges",hue="region",data=df)
plt.show()


# In[22]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# In[23]:


sns.scatterplot(x="bmi",y="charges",hue="smoker",data=df)
plt.show()


# In[24]:


sns.scatterplot(x="bmi",y="charges",hue="sex",data=df)
plt.show()


# In[25]:


sns.scatterplot(x="children",y="charges",hue="smoker",data=df)
plt.show()


# In[26]:


df.head()


# In[27]:


from sklearn.preprocessing import LabelEncoder
L1=LabelEncoder()
L2=LabelEncoder()
L3=LabelEncoder()
df["sex"]=L1.fit_transform(df['sex'])
df["smoker"]=L2.fit_transform(df['smoker'])
df["region"]=L3.fit_transform(df['region'])


# In[28]:


print(L1.classes_)
print(L2.classes_)
print(L3.classes_)


# In[29]:


df.head()


# In[30]:


x=df.drop('charges',axis=1)
y=df['charges']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[31]:


print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)


# In[32]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor


# In[33]:


algo=MLPRegressor(hidden_layer_sizes=(50,50),max_iter=2000,verbose=True)
algo.fit(xtrain,ytrain)


# In[34]:


ypred=algo.predict(xtest)


# In[35]:


from sklearn.metrics import mean_absolute_error,r2_score
print(mean_absolute_error(ytest,ypred))
print(r2_score(ytest,ypred))


# In[36]:


algo2=LinearRegression()
algo2.fit(xtrain,ytrain)


# In[37]:


ypred=algo2.predict(xtest)


# In[38]:


print(mean_absolute_error(ytest,ypred))
print(r2_score(ytest,ypred))


# In[39]:


algo3=DecisionTreeRegressor()
algo3.fit(xtrain,ytrain)


# In[40]:


ypred=algo3.predict(xtest)
print(mean_absolute_error(ytest,ypred))
print(r2_score(ytest,ypred))


# In[42]:


import joblib
joblib.dump(algo,r'C:\Users\aios210720\Downloads\insurance.csv')


# In[43]:


new=np.array([[25,0,38.23,3,1,1]])
algo.predict(new)


# In[45]:


joblib.dump(L1,r'C:\Users\aios210720\Downloads\insurance.csv')
joblib.dump(L2,r'C:\Users\aios210720\Downloads\insurance.csv')
joblib.dump(L3,r'C:\Users\aios210720\Downloads\insurance.csv')


# In[ ]:




