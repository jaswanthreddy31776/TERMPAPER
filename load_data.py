import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# PHASE 4: DATA PREPROCESSING - CUSTOMIZED FOR YOUR CSV

# Step 1: Load your dataset
df = pd.read_csv("student-scores.csv")

# Step 2: View the first few rows
print("🔹 First 5 Rows:")
print(df.head())

# Step 3: Handle missing values
df.fillna(0, inplace=True)

# Step 4: Print column names
print("\n🔹 Columns in your dataset:")
print(df.columns)

# Step 5: Create quiz_score = average of all subject scores
subject_cols = [
    'math_score', 'history_score', 'physics_score', 'chemistry_score',
    'biology_score', 'english_score', 'geography_score'
]

df['quiz_score'] = df[subject_cols].mean(axis=1)

# Step 6: Simulate time_spent column
np.random.seed(42)
df['time_spent'] = np.random.randint(5, 60, size=len(df))  # minutes per week

# Step 7: Normalize quiz_score and time_spent
scaler = MinMaxScaler()
df[['quiz_score', 'time_spent']] = scaler.fit_transform(df[['quiz_score', 'time_spent']])

# Step 8: Calculate engagement_score
df['engagement_score'] = df['quiz_score'] * 0.6 + df['time_spent'] * 0.4

# Step 9: Preview the result
print("\n🔹 Processed Data (first 5 rows):")
print(df[['quiz_score', 'time_spent', 'engagement_score']].head())

# Step 10: Save processed data
df.to_csv("processed_students.csv", index=False)

# PHASE 5: DATA ANALYSIS & VISUALIZATION

# Step 1: Visualize Quiz Score Distribution
sns.histplot(df['quiz_score'], kde=True, color='skyblue')
plt.title("Quiz Score Distribution")
plt.xlabel("Quiz Score")
plt.ylabel("Number of Students")
plt.show()

# Step 2: Visualize Time Spent Distribution
sns.histplot(df['time_spent'], kde=True, color='salmon')
plt.title("Time Spent Distribution")
plt.xlabel("Time Spent (Normalized)")
plt.ylabel("Number of Students")
plt.show()

# Step 3: Engagement Score vs Quiz Score (Scatter Plot)
plt.figure(figsize=(8,6))
sns.scatterplot(x='quiz_score', y='engagement_score', data=df)
plt.title("Engagement Score vs Quiz Score")
plt.xlabel("Quiz Score")
plt.ylabel("Engagement Score")
plt.show()

# Step 4: Box Plot for All Subjects (Optional)
subject_cols = [
    'math_score', 'history_score', 'physics_score',
    'chemistry_score', 'biology_score', 'english_score', 'geography_score'
]

df_melted = df.melt(value_vars=subject_cols, var_name='Subject', value_name='Score')
sns.boxplot(x='Subject', y='Score', data=df_melted)
plt.title("Score Distribution by Subject")
plt.xticks(rotation=45)
plt.show()


# PHASE 6: MACHINE LEARNING MODELS

# Step 1: KMeans Clustering Based on Engagement Score
# Select the relevant columns for clustering
X = df[['quiz_score', 'time_spent', 'engagement_score']]

# Initialize the KMeans model and fit to the data
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Step 2: Visualize Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='quiz_score', y='engagement_score', hue='cluster', data=df, palette='viridis', s=100)
plt.title("Clustering Students Based on Engagement Score")
plt.xlabel("Quiz Score")
plt.ylabel("Engagement Score")
plt.legend(title='Cluster')
plt.show()

# Step 3: Recommending Learning Materials Based on Cluster
# Define learning materials for each cluster (hypothetical recommendations)
learning_materials = {
    0: ["Beginner Math Resources", "Intro to History", "Basic Physics Notes"],
    1: ["Advanced Math Challenges", "Research in Physics", "Deep Dive into Chemistry"],
    2: ["Math Puzzles", "Geography Expeditions", "Biology Experiments"]
}

# Add a column for recommended materials
df['recommended_materials'] = df['cluster'].apply(lambda x: learning_materials[x])

# Step 4: Display Recommendations for Each Student
# Check if 'student_id' exists in the DataFrame, otherwise skip it
if 'student_id' in df.columns:
    print("\n🔹 Learning Material Recommendations Based on Clusters:")
    print(df[['student_id', 'cluster', 'recommended_materials']].head())
else:
    print("\n🔹 'student_id' column not found. Showing recommendations without student_id:")
    print(df[['cluster', 'recommended_materials']].head())

# Step 5: Save the final dataset with recommendations
df.to_csv("students_with_recommendations.csv", index=False)
print("File 'students_with_recommendations.csv' saved successfully!")

# PHASE 7: MODEL EVALUATION

# Step 1: Evaluate Model Performance
# Calculate Inertia (within-cluster sum of squares)
print("\n🔹 Inertia (Within-cluster Sum of Squares):", kmeans.inertia_)

# Step 2: Calculate Silhouette Score
silhouette_avg = silhouette_score(X, df['cluster'])
print("🔹 Silhouette Score:", silhouette_avg)

# Step 3: Evaluate the number of clusters using Elbow Method (Optional)
inertia_values = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Plot the Elbow graph
plt.plot(range(1, 11), inertia_values)
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
# Standardize the features
X_scaled = StandardScaler().fit_transform(df[['quiz_score', 'time_spent', 'engagement_score']])

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

# Visualize
sns.scatterplot(data=df, x='quiz_score', y='engagement_score', hue='dbscan_cluster', palette='tab10')
plt.title("DBSCAN Clustering")
plt.show()
Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Student Index")
plt.ylabel("Distance")
plt.show()
