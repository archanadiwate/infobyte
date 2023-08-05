#!/usr/bin/env python
# coding: utf-8

# # OASIS INFOBYTE DATA SCIENCE INTERNSHIP 2023

# # Task 2 UNEMPLOYMENT ANALYSIS WITH PYTHON
#    Unemployment is measured by the unemployment rate which is the number of people who are unemployed as a percentage of the total labour force. We have seen a sharp increase in the unemployment rate during Covid-19, so analyzing the unemployment rate can be a good data science project.

# # Name- Archana Diwate

# # 1. Import the libraries

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# # 2. Import the dataset

# In[15]:


df=pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
df


# # 3. Analysis

# In[16]:


df.head(100)


# In[19]:


#Checking for missing values
df.isnull().sum()


# In[18]:


#correlation between the features of this dataset
df.corr()


# # 4.Visualization

# In[21]:


#visual representation of correlation between the features of this dataset

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(14, 12))
sns.heatmap(df.corr())
plt.show()


# In[22]:


#unemployment rate according to different regions of India
df.columns= ["States","Date","Frequency",
               "Estimated Unemployment Rate","Estimated Employed",
               "Estimated Labour Participation Rate","Region",
               "longitude","latitude"]

plt.figure(figsize=(10, 8))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate", hue="Region", data=df)
plt.show()


# In[23]:


sns.pairplot(df)


# In[24]:


unemploment = df[["States", "Region", "Estimated Unemployment Rate"]]
figure = px.sunburst(unemploment, path=["Region", "States"], 
                     values="Estimated Unemployment Rate", 
                     width=500, height=500, color_continuous_scale="RdY1Gn", 
                     title="Unemployment Rate in India")
figure.show()


# # Summary
# Analysis of unemployment rate by using the python programming language is done.
