#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings 

warnings.filterwarnings("ignore")


# In[4]:


haberman = pd.read_csv(r'C:\Users\aios210720\Downloads\haberman.csv')


# In[5]:


haberman.columns = ['age','year','nodes','status']
haberman


# In[6]:


haberman.shape


# In[7]:


print(haberman.columns)


# In[8]:


haberman.info()


# In[9]:


haberman.describe()


# In[12]:


haberman["status"].value_counts()


# In[13]:


survive= haberman.loc[haberman["status"] == 1]
unsurvive = haberman.loc[haberman["status"] == 2]
plt.title("1-D Scatter plot for detcteted age and status ")
plt.plot(survive["age"], np.zeros_like(survive['age']), 'o',label ='Survived')
plt.plot(unsurvive["age"], np.zeros_like(unsurvive['age']), 'o',label ='Unsurvived')
plt.xlabel('Age')
plt.legend()
plt.show()


# In[14]:


haberman.plot(kind='scatter', x='age', y='year',label="Age")
plt.title("2-D Scatter plot for detcteted age and year ")
plt.legend()
plt.show()


# In[20]:


sns.set_style("whitegrid")
sns.FacetGrid(haberman, hue="status", size=5)    .map(plt.scatter, "age","year")    .add_legend()
plt.title("2-D Scatter plot for detcteted age and year base on status")
plt.show();


# In[22]:


plt.close()
sns.set_style("whitegrid")
titl=sns.pairplot(haberman, hue="status", vars=['age','nodes','year'],size=4)
titl.fig.suptitle("Pairplot of Age, Nodes & Year")
plt.show()


# In[23]:


plt.hist(haberman["age"],label="Age")
plt.legend()
plt.title("Histogram plot for detcteted age.")


# In[24]:


sns.FacetGrid(haberman, hue="status", size=5)    .map(sns.distplot, "age")    .add_legend()
plt.title("PDF plot for detcteted age & Status.")
plt.show()


# In[25]:


sns.FacetGrid(haberman, hue="status", size=5)    .map(sns.distplot, "nodes")    .add_legend()
plt.title("PDF plot for detcteted Nodes & Status.")
plt.show()


# In[26]:


sns.FacetGrid(haberman, hue="status", size=5)    .map(sns.distplot, "year")    .add_legend()
plt.title("PDF plot for detcteted Year & Status.")
plt.show()


# In[27]:


counts, bin_edges = np.histogram(survive['age'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.title("CDF plot for detcteted age.")
plt.plot(bin_edges[1:],pdf,label="PDF survived")
plt.plot(bin_edges[1:], cdf,label="CDF survived")
plt.legend() 


# In[28]:


counts, bin_edges = np.histogram(survive['year'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.title("CDF plot for detcteted Year.")
plt.plot(bin_edges[1:],pdf,label="PDF survived")
plt.plot(bin_edges[1:], cdf,label="CDF survived")
plt.legend()


# In[29]:


counts, bin_edges = np.histogram(unsurvive['nodes'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.title("CDF plot for detcteted Nodes base on Unsurvive.")
plt.plot(bin_edges[1:],pdf,label="PDF unsurvived")
plt.plot(bin_edges[1:], cdf,label="CDF unsurvived")
plt.legend()


# In[30]:


print("Means:")
print(np.mean(haberman["nodes"]))
print(np.mean(haberman["age"]))

print("\nVariance:")
print(np.var(haberman['nodes']))
print(np.var(haberman['age']))

print("\nStd-dev:")
print(np.std(haberman["nodes"]))
print(np.std(haberman["age"]))


# In[31]:


print("\nMedians:")
print(np.median(haberman["age"]))

print("\nQuantiles:")
print(np.percentile(haberman["age"],np.arange(0, 100, 25)))

print("\n90th Percentiles:")
print(np.percentile(haberman["age"],90))


from statsmodels import robust
print("\nMedian Absolute Deviation:")
print(robust.mad(haberman["age"]))


# In[32]:


sns.boxplot(x='status',y='age', data=haberman)
plt.title("Box and Whiskers plot for detcteted Age & Status")
plt.show()


# In[33]:


sns.boxplot(x='status',y='year', data=haberman)
plt.title("Box and Whiskers plot for detcteted year & Status")
plt.show()


# In[34]:


sns.violinplot(x="status", y="age", data=haberman, size=8)
plt.title("1) Violin plot for detcteted age and survival status")
plt.legend(labels=["status","age"])
plt.show()


# In[35]:


sns.violinplot(x="status", y="year", data=haberman, size=8)
plt.title("3) Violin plot for detcteted year and survival status ")
plt.legend(labels=["status","year"])
plt.show()


# In[36]:


sns.violinplot(x="status", y="nodes", data=haberman, size=8)
plt.title("2) Violin plot for detcteted nodes and survival status ")
plt.legend(labels=["status","nodes"])
plt.show()


# In[37]:


sns.jointplot(x=haberman.year, y=haberman.nodes, kind="kde")
plt.legend(labels=["nodes"])
plt.show()


# In[38]:


sns.set_style("white")
plt.title("contour plot for detcteted Age & year")
sns.kdeplot(x=haberman.age, y=haberman.year, cmap="Blues", shade=True, thresh=0)
plt.show()


# In[ ]:




