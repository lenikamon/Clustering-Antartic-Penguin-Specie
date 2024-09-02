#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
penguins_df.head()
penguins_df.info()
penguins_df.describe()

# drop missing values
penguins_df = penguins_df.dropna()

# Convert categorical variables into dummy/indicator variables
penguins_df = pd.get_dummies(penguins_df, dtype='int') 

#numeric columns for clustering
numeric_df = penguins_df[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# Fit the KMeans model with the chosen number of clusters
optimal_clusters = 4 
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
penguins_df['cluster_labels'] = kmeans.fit_predict(scaled_data)

# Calculate mean values for each cluster
stat_penguins = penguins_df.groupby('cluster_labels').mean()

print(stat_penguins)
penguins_df.head()

