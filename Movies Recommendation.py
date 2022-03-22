#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[2]:


import pandas as pd
import numpy as np


# # import the movies dataset

# In[3]:


movie_df = pd.read_csv(r'C:\Users\aios210720\Downloads\movies.csv')


# In[4]:


rating_df = pd.read_csv(r'C:\Users\aios210720\Downloads\ratings.csv')


# # checking the tables

# In[5]:


movie_df.head(10)


# In[6]:


rating_df.head(10)


# In[7]:


combine_movie_rating = pd.merge(rating_df, movie_df, on='movieId')
combine_movie_rating.head(10)


# In[8]:


columns = ['timestamp', 'genres']
combine_movie_rating = combine_movie_rating.drop(columns, axis=1)
combine_movie_rating.head(10)


# In[9]:


combine_movie_rating = combine_movie_rating.dropna(axis = 0, subset = ['title'])

movie_ratingCount = (combine_movie_rating.
     groupby(by = ['title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['title', 'totalRatingCount']]
    )
movie_ratingCount.head(10)


# In[10]:


rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
rating_with_totalRatingCount.head(10)


# In[11]:


user_rating = rating_with_totalRatingCount.drop_duplicates(['userId','title'])
user_rating.head(10)


# # matrix factorization

# In[12]:


movie_user_rating_pivot = user_rating.pivot(index = 'userId', columns = 'title', values = 'rating').fillna(0)
movie_user_rating_pivot.head(10)


# In[13]:


X = movie_user_rating_pivot.values.T
X.shape


# In[14]:


import sklearn
from sklearn.decomposition import TruncatedSVD

SVD = TruncatedSVD(n_components=12, random_state=17)
matrix = SVD.fit_transform(X)
matrix.shape


# In[15]:


import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
corr = np.corrcoef(matrix)
corr.shape


# # now check the result

# In[16]:


movie_title = movie_user_rating_pivot.columns
movie_title_list = list(movie_title)
coffey_hands = movie_title_list.index("Guardians of the Galaxy (2014)")


# In[17]:


corr_coffey_hands  = corr[coffey_hands]
list(movie_title[(corr_coffey_hands >= 0.9)])


# In[ ]:




