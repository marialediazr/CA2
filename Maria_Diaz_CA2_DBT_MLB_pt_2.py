#!/usr/bin/env python
# coding: utf-8

# ## Data Visualisation
# 
# #### Clustering
# - Create an interactive Dashboard aimed at younger adults (18 - 35 years) with specific features to summarise the most important aspects of the data and identify through your visualisation why this dataset is suitable for Machine Learning models in an online retail business. Explain how your dashboard is designed with this demographic in mind. 
# 
# - Discuss in detail your rationale and justification for all stages of data preparation for your visualizations.
# 
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import cufflinks
from matplotlib import gridspec
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
from matplotlib.colors import ListedColormap
import streamlit as st


# In[2]:


st.title("Movies Dashboard")


# In[2]:


with open('movies.csv', 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    print(result) 


# In[3]:


movies = pd.read_csv('movies.csv', encoding='ISO-8859-1')

movies.to_csv('movies_utf8.csv', index=False, encoding='utf-8')


# In[4]:


movies.head(15)


# In[5]:


movies.info()


# In[6]:


movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
movies.head()


# In[7]:


unique_value_counts_movies = movies.nunique()
print(unique_value_counts_movies)


# In[8]:


sorted_years = sorted(movies["year"].unique(), reverse=True)
print(sorted_years)


# In[9]:


ratings = pd.read_csv("ratings.csv")


# In[10]:


ratings.head()


# In[11]:


ratings.tail()


# In[12]:


ratings.shape


# In[13]:


ratings.info()


# In[14]:


for column in ratings.columns:
    unique_values = ratings[column].unique()
    sorted_values = sorted(unique_values, key=lambda x: (isinstance(x, str), x))
    print(f"Unique values of {column}: {sorted_values}")
    print()


# In[15]:


unique_value_counts_ratings = ratings.nunique()
print(unique_value_counts_ratings)


# In[16]:


with open('tags.csv', 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    print(result) 


# In[17]:


tags = pd.read_csv('tags.csv', encoding='Windows-1252')

tags.to_csv('tags_utf8.csv', index=False, encoding='utf-8')


# In[18]:


tags.head()


# In[19]:


tags.info()


# In[20]:


unique_value_counts_tags = tags.nunique()
print(unique_value_counts_tags)


# In[21]:


unique_values_list_tags = tags['tag'].unique().tolist()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(unique_values_list_tags)


# In[22]:


average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
average_ratings.columns = ['movieId', 'average_rating']
average_ratings['average_rating'] = average_ratings['average_rating'].round(1)
print(average_ratings)


# def round_to_half(value):
#     return round(value * 2) / 2
# average_ratings['average_rating'] = average_ratings['average_rating'].apply(round_to_half)
# print(average_ratings)

# In[23]:


movies2 = movies.copy()


# In[24]:


movies2 = movies2.merge(average_ratings, on='movieId', how='left')


# In[25]:


movies2.head()


# In[26]:


movies2.describe()


# In[27]:


movies_graphs = movies2.copy()


# In[28]:


movies2.drop('title', axis=1, inplace=True)


# In[29]:


movies2.head()


# In[30]:


movies2.info()


# In[31]:


movies2['year'] = movies2['year'].astype(int)


# In[32]:


movies_per_year = movies2.groupby('year').size()

plt.figure(figsize=(10, 6))
plt.plot(movies_per_year.index, movies_per_year.values, marker='o')
plt.title('Number of Movies Released Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')

custom_ticks = list(range(movies_per_year.index.min(), movies_per_year.index.max() + 1, 5))
plt.xticks(custom_ticks)
plt.xticks(rotation=90, fontsize=8)

plt.grid(True)
plt.show()


# In[33]:


yearly_avg_rating = movies2.groupby('year')['average_rating'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(yearly_avg_rating['year'], yearly_avg_rating['average_rating'], marker='o')
plt.title('Year vs Average Rating', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)

x_ticks = range(yearly_avg_rating['year'].min(), yearly_avg_rating['year'].max() + 1, 5)
plt.xticks(x_ticks, rotation=90, fontsize=10)

plt.grid(axis='y')
plt.tight_layout()

plt.show()


# encoder = preproc.LabelEncoder()
# movies2['genres'] = encoder.fit_transform(movies2['genres'])
# mapping = dict(enumerate(encoder.classes_))
# print("Mapping of encoded numbers to original genres:")
# for key, value in mapping.items():
#     print(f"{key}: {value}")

# In[34]:


movies_graphs['year'] = movies_graphs['year'].astype(int)

fig = px.scatter(movies_graphs,
    x='year',
    y='average_rating',
    hover_data=['title'],
    title='Year vs Average Rating',
    labels={'year': 'Year', 'average_rating': 'Average Rating'},)

fig.update_layout(xaxis=dict(
        tickmode='linear',
        tick0=1990,
        dtick=5,
        title='Year',),
    yaxis=dict(title='Average Rating',),
    hoverlabel=dict(font_size=12))

fig.show()


# In[35]:


movies2.head()


# In[36]:


genre_counts = movies2['genres'].value_counts()
print(genre_counts)

