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


get_ipython().system('pip install streamlit pandas plotly')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import seaborn as sns
sns.set(color_codes=True)
import chart_studio.plotly as py
import plotly.graph_objs as go
import chardet
from plotly.offline import iplot, init_notebook_mode
import cufflinks
from matplotlib import gridspec
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
from matplotlib.colors import ListedColormap


# In[3]:


with open('movies.csv', 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    print(result) 


# In[4]:


movies = pd.read_csv('movies.csv', encoding='ISO-8859-1')

movies.to_csv('movies_utf8.csv', index=False, encoding='utf-8')


# In[5]:


movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
movies.head()


# In[6]:


ratings = pd.read_csv("ratings.csv")


# In[7]:


ratings.head()


# In[8]:


ratings.tail()


# In[9]:


with open('tags.csv', 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    print(result) 


# In[10]:


tags = pd.read_csv('tags.csv', encoding='Windows-1252')

tags.to_csv('tags_utf8.csv', index=False, encoding='utf-8')


# In[11]:


tags.head()


# In[12]:


average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
average_ratings.columns = ['movieId', 'average_rating']
average_ratings['average_rating'] = average_ratings['average_rating'].round(1)
print(average_ratings)


# def round_to_half(value):
#     return round(value * 2) / 2
# average_ratings['average_rating'] = average_ratings['average_rating'].apply(round_to_half)
# print(average_ratings)

# In[13]:


movies2 = movies.copy()


# In[14]:


movies2 = movies2.merge(average_ratings, on='movieId', how='left')


# In[15]:


movies_graphs = movies2.copy()


# In[16]:


movies2.drop('title', axis=1, inplace=True)


# In[17]:


movies2['year'] = movies2['year'].astype(int)


# encoder = preproc.LabelEncoder()
# movies2['genres'] = encoder.fit_transform(movies2['genres'])
# mapping = dict(enumerate(encoder.classes_))
# print("Mapping of encoded numbers to original genres:")
# for key, value in mapping.items():
#     print(f"{key}: {value}")

# In[18]:


st.set_page_config(
    page_title="Movie Analytics Dashboard",
    layout="wide"
)

st.title("Movie Analytics Dashboard")

# Visualization 1: Top 20 Movies by Rating
st.header("1. Top 20 Movies by Rating")
st.write("This visualization shows the top 20 movies of all time, sorted by their average rating.")
top_movies = movies_graphs.sort_values(by='average_rating', ascending=False).head(20)
fig_top_movies = px.bar(
    top_movies,
    x='average_rating',
    y='title',
    orientation='h',
    color='genres',
    labels={'average_rating': 'Average Rating', 'title': 'Movie Title', 'genres': 'Genre'},
    title='Top 20 Movies of All Time Based on Average Rating',
    color_discrete_sequence=px.colors.qualitative.Bold
)
st.plotly_chart(fig_top_movies, use_container_width=True)

# Visualization 2: Number of Movies per Genre
st.header("2. Number of Movies per Genre")
st.write("This visualization shows the total number of movies in each genre.")
df_genres = movies_graphs.copy()
df_genres['genres'] = df_genres['genres'].str.split('|')
df_genres = df_genres.explode('genres')
genre_counts = df_genres['genres'].value_counts().reset_index()
genre_counts.columns = ['Genre', 'Number of Movies']
fig_genres = px.bar(
    genre_counts,
    x="Genre",
    y="Number of Movies",
    title="Number of Movies per Genre",
    labels={"Genre": "Movie Genre", "Number of Movies": "Number of Movies"},
    color="Genre",
    color_discrete_sequence=px.colors.qualitative.Bold
)
st.plotly_chart(fig_genres, use_container_width=True)

# Visualization 3: Year vs Average Rating
st.header("3. Year vs Average Rating")
st.write("This scatter plot shows the relationship between the release year of movies and their average rating.")
movies_graphs['year'] = movies_graphs['year'].astype(int)
fig_year_avg = px.scatter(
    movies_graphs,
    x='year',
    y='average_rating',
    color='genres',
    hover_data=['title'],
    title='Year vs Average Rating',
    labels={'year': 'Year', 'average_rating': 'Average Rating', 'genres': 'Genre'},
    color_discrete_sequence=px.colors.qualitative.Bold
)
st.plotly_chart(fig_year_avg, use_container_width=True)

# Visualization 4, 5, and 6
# Add placeholders for your remaining visualizations and their descriptions
# Replace these with actual data processing and plotting code
st.header("4. Visualization 4 Title")
st.write("Description for the fourth visualization.")
st.plotly_chart(px.scatter(), use_container_width=True)  # Placeholder

st.header("5. Visualization 5 Title")
st.write("Description for the fifth visualization.")
st.plotly_chart(px.bar(), use_container_width=True)  # Placeholder

st.header("6. Visualization 6 Title")
st.write("Description for the sixth visualization.")
st.plotly_chart(px.line(), use_container_width=True)  # Placeholder


# In[22]:


jupyter nbconvert --to script dashboard.ipynb

