#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-05/villagers.csv"
df = pd.read_csv(url)
df.isna().sum()


# In[6]:


rows, columns = df.shape
print(f'Number of rows: {rows}')
print(f'Number of columns: {columns}')
#observations are individual entries and each rows represent an observation.variables are each piece of informarion collected and each column represents a variable.


# In[9]:


df.describe()


# In[11]:


df.shape
#df.describe() only shows the numeric columns while df.shape shows all the rows and the columns. Hence, df.describe usually has less row than the numeric columns.
#the 'count' column represents how many not-null columns there are.


# In[ ]:


#1. attribute does not have parenthese while method contains the parathese.
#2.attribute represents just the data and prpoerty of the dataset, but the method is an action that can be performed.
#3 attribute often usaully provide the value and property  directly, but the method often performs operations and return a result.


# In[33]:


#when i want to remove some rows or columns with missing value so that i can ensure the completeness of the data, i should use the df.dropna().
#when a whole column is no longer needed or i want to remove a column regardless of whether it contains missing value or not, i should use def df['col']
#Applying 'def df['col']' before the df.dropna() is important because it can enhance the efficiency, focus on relevant data and avoid errors. It can also lead to more acuurate perfoemance.
del df['song']# I think song is an irrelevant column, so i delate it firstly so that it will be more efficient
df.dropna()#After removing irrelevant column, i can delate those missing values.


# In[ ]:


#count is the number of the observations.mean is the average value of the numbers. 50% means medium value. min means the smallest value and max means the largest number.
#25% means the value below which 25% of the data falls and 75% means the value below 75% of the data falls.std means the spread of the value around the number.


# In[42]:


import pandas as pd

# Load the dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
df = pd.read_csv(url)
df.isna().sum()
df.groupby("class")["age"].describe()


# In[ ]:


#df.groupby groups the DataFrame by the unique values in the column


# In[ ]:


url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
df = pd.read_csv(url)
df.isna().sum()


# In[43]:


titanics.csv


# In[ ]:


DF.groupby("class")["age"].describe()


# In[ ]:


pd.read_csv(url


# In[ ]:


df.group_by("class")["age"].describe()


# In[ ]:


titanic_df.groupby("sex")["Age"].describe()


# In[ ]:


titanic_df.groupby("sex")[age].describe()


# In[ ]:


#no

