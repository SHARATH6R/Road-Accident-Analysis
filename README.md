# Install necessary libraries
!pip install pandas matplotlib seaborn folium scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.cluster import KMeans

# Step 1: Load the dataset
from google.colab import files
uploaded = files.upload()

data = pd.read_csv("road_accidents.csv")  # Adjust file name if needed
print(data.head())

# Step 2: Basic Exploration
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Step 3: Analyze data by weather, road conditions, and time of day
# Example 1: Weather conditions
plt.figure(figsize=(10, 6))
sns.countplot(y='Weather_Condition', data=data, order=data['Weather_Condition'].value_counts().index[:10])
plt.title('Top 10 Weather Conditions in Accidents')
plt.show()

# Example 2: Road conditions
plt.figure(figsize=(10, 6))
sns.countplot(y='Road_Condition', data=data, order=data['Road_Condition'].value_counts().index[:10])
plt.title('Top 10 Road Conditions in Accidents')
plt.show()

# Example 3: Time of Day
data['Hour'] = pd.to_datetime(data['Start_Time']).dt.hour
plt.figure(figsize=(10, 6))
sns.histplot(data['Hour'], bins=24, kde=False)
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Accidents')
plt.show()

# Step 4: Identify Accident Hotspots
# Ensure Latitude and Longitude columns are present
data = data.dropna(subset=['Start_Lat', 'Start_Lng'])
map = folium.Map(location=[data['Start_Lat'].mean(), data['Start_Lng'].mean()], zoom_start=5)

# Plot accident locations
for _, row in data.sample(1000).iterrows():  # Sample 1000 for performance
    folium.CircleMarker(location=[row['Start_Lat'], row['Start_Lng']], radius=2, color='red').add_to(map)

map

# Step 5: Clustering Accident Hotspots
kmeans = KMeans(n_clusters=10)
data['Cluster'] = kmeans.fit_predict(data[['Start_Lat', 'Start_Lng']])
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Start_Lng', y='Start_Lat', hue='Cluster', data=data, palette='viridis')
plt.title('Accident Clusters')
plt.show()

# Step 6: Save processed data if needed
data.to_csv('processed_accident_data.csv', index=False)
