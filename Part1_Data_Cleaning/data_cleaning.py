#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd


# # Part 1: Data Cleaning

# In[27]:


# Loading data
data = pd.read_csv("/Volumes/Data/Data Science Bootcamp/Assignments/ML OPS/heart.csv")


# In[29]:


data.head()


# In[31]:


data.shape


# In[33]:


data.isnull().sum()


# ### There are no missing values in data

# #### Encoding Categorical Variables

# In[35]:


from sklearn.preprocessing import LabelEncoder

categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Initialize the LabelEncoder
encoder = LabelEncoder()

# Function to apply LabelEncoder to multiple columns
def encode_columns(df, columns):
    for column in columns:
        df[column] = encoder.fit_transform(df[column])
    return df

# Apply Label encoding
cleaned_data = encode_columns(data, categorical_cols)

cleaned_data.head()


# In[37]:


# Save the cleaned data to a CSV file
cleaned_data.to_csv('cleaned_heart_disease_data.csv', index=False)

