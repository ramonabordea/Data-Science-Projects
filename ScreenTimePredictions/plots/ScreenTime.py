import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Set style
plt.style.use('default')

# Load the dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'user_behavior_dataset.csv')
df = pd.read_csv(csv_path)
print(df.columns)

# 1. Gender Distribution
plt.figure(figsize=(10, 6))
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
plt.title('Gender Distribution')
#plt.savefig('1_gender_distribution.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

print("\nGender Distribution:")
print(gender_counts)

# 2. Device Models Distribution
plt.figure(figsize=(12, 6))
device_counts = df['Device Model'].value_counts().head(10)
sns.barplot(x=device_counts.values, y=device_counts.index)
plt.title('Top 10 Device Models')
plt.xlabel('Count')
plt.tight_layout()
#plt.savefig('2_device_models_distribution.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# 3. Apps vs Screen Time Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Number of Apps Installed', y='Screen On Time (hours/day)')
plt.title('Number of Apps vs Screen Time')
plt.xlabel('Number of Apps Installed')
plt.ylabel('Screen Time (hours/day)')
#plt.savefig('3_apps_vs_screentime.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

correlation = df['Number of Apps Installed'].corr(df['Screen On Time (hours/day)'])
print(f"\nCorrelation between Apps Installed and Screen Time: {correlation:.2f}")

# 4. Screen Time by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Gender', y='Screen On Time (hours/day)')
plt.title('Screen Time Distribution by Gender')
plt.ylabel('Screen Time (hours/day)')
#plt.savefig('4_screentime_by_gender.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

print("\nAverage Screen Time by Gender:")
print(df.groupby('Gender')['Screen On Time (hours/day)'].mean())

# 5. Age and Screen Time Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Screen On Time (hours/day)', hue='Gender', alpha=0.6)
plt.title('Screen Time vs Age (Colored by Gender)')
plt.xlabel('Age')
plt.ylabel('Screen Time (hours/day)')
#plt.savefig('5_age_screentime_gender.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# 6. App Usage Time vs Screen Time
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='App Usage Time (min/day)', y='Screen On Time (hours/day)')
plt.title('App Usage Time vs Screen Time')
plt.xlabel('App Usage Time (minutes/day)')
plt.ylabel('Screen Time (hours/day)')
#plt.savefig('6_app_usage_vs_screentime.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# 7. User Behavior Class Distribution
plt.figure(figsize=(10, 6))
behavior_counts = df['User Behavior Class'].value_counts()
sns.barplot(x=behavior_counts.index, y=behavior_counts.values)
plt.title('Distribution of User Behavior Classes')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.savefig('7_user_behavior_distribution.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# 8. Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
#plt.savefig('8_age_distribution.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# 9. Correlation Matrix
numeric_cols = ['Age', 'Screen On Time (hours/day)', 'App Usage Time (min/day)', 
                'Battery Drain (mAh/day)', 'Number of Apps Installed', 'Data Usage (MB/day)']
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
#plt.savefig('9_correlation_matrix.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# Save summary statistics and insights
with open('analysis_results.txt', 'w') as f:
    f.write("Data Analysis Results\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. Gender Distribution:\n")
    f.write(str(gender_counts) + "\n\n")
    
    f.write("2. Screen Time Statistics by Gender:\n")
    f.write(str(df.groupby('Gender')['Screen On Time (hours/day)'].describe()) + "\n\n")
    
    f.write("3. Correlation Analysis:\n")
    f.write(f"Apps vs Screen Time correlation: {correlation:.2f}\n\n")
    
    f.write("4. Age Statistics:\n")
    f.write(str(df['Age'].describe()) + "\n\n")
    
    f.write("5. User Behavior Class Distribution:\n")
    f.write(str(behavior_counts) + "\n\n")

print("\nAnalysis complete! Check the current directory for:")
print("- 9 visualization PNG files")
print("- analysis_results.txt with detailed statistics")

# Add this after your existing EDA code
print("\n" + "="*50)
print("Random Forest Model Analysis")
print("="*50)

# Prepare features (X) and target (y)
X = df.drop(['Screen On Time (hours/day)', 'User ID', 'Device Model', 
             'Operating System', 'Gender', 'User Behavior Class'], axis=1)
y = df['Screen On Time (hours/day)']

# Print the features being used
print("\nFeatures used for prediction:")
print("-" * 50)
for i, feature in enumerate(X.columns, 1):
    print(f"{i}. {feature}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Create and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print model performance
print("\nModel Performance Metrics:")
print("-" * 50)
print(f"R-squared Score: {r2:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Feature importance plot
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Predicting Screen Time')
plt.savefig('10_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Prediction scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Screen Time')
plt.ylabel('Predicted Screen Time')
plt.title('Actual vs Predicted Screen Time')
plt.savefig('11_prediction_scatter.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Residual plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted Screen Time')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.savefig('12_residual_plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Save model results
with open('random_forest_results.txt', 'w') as f:
    f.write("Random Forest Model Results\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. Model Performance Metrics:\n")
    f.write(f"R-squared Score: {r2:.4f}\n")
    f.write(f"Root Mean Squared Error: {rmse:.4f}\n")
    f.write(f"Mean Absolute Error: {mae:.4f}\n\n")
    
    f.write("2. Feature Importance:\n")
    for idx, row in feature_importance.iterrows():
        f.write(f"{row['feature']}: {row['importance']:.4f}\n")
    
    f.write("\n3. Dataset Split:\n")
    f.write(f"Training set size: {len(X_train)}\n")
    f.write(f"Testing set size: {len(X_test)}\n")

print("\nRandom Forest Analysis complete! Additional files created:")
print("- 10_feature_importance.png")
print("- 11_prediction_scatter.png")
print("- 12_residual_plot.png")
print("- random_forest_results.txt")

# ------------------------------------------------------------------------------------------------------------

print("\nClustering Analysis")
print("="*50)

# Prepare data for clustering
features_for_clustering = [
    'App Usage Time (min/day)',
    'Screen On Time (hours/day)',
    'Battery Drain (mAh/day)',
    'Number of Apps Installed',
    'Data Usage (MB/day)',
    'Age'
]

X_cluster = df[features_for_clustering]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
X_scaled = pd.DataFrame(X_scaled, columns=features_for_clustering)

# K-means Clustering - Find optimal k
inertias = []
silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'rx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.tight_layout()
plt.savefig('13_kmeans_optimal_k.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Perform K-means with optimal k
optimal_k = 4  # Based on elbow curve and silhouette score
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_scaled['Screen On Time (hours/day)'], 
                     X_scaled['App Usage Time (min/day)'],
                     c=df['Cluster'], 
                     cmap='viridis')
plt.xlabel('Standardized Screen Time')
plt.ylabel('Standardized App Usage Time')
plt.title('K-means Clusters')
plt.colorbar(scatter)
plt.savefig('14_kmeans_clusters.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Hierarchical Clustering
linkage_matrix = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.savefig('15_hierarchical_clustering.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Analyze Clusters
cluster_means = df.groupby('Cluster')[features_for_clustering].mean()

# Plot cluster characteristics
plt.figure(figsize=(15, 8))
cluster_means_scaled = cluster_means.copy()
for column in cluster_means_scaled.columns:
    cluster_means_scaled[column] = (cluster_means_scaled[column] - df[column].mean()) / df[column].std()

sns.heatmap(cluster_means_scaled, annot=True, cmap='coolwarm', center=0)
plt.title('Cluster Characteristics (Standardized Values)')
plt.savefig('16_cluster_characteristics.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Save clustering results
with open('clustering_results.txt', 'w') as f:
    f.write("Clustering Analysis Results\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. K-means Clustering\n")
    f.write(f"Optimal number of clusters (k): {optimal_k}\n")
    f.write(f"Silhouette score: {silhouette_scores[optimal_k-2]:.4f}\n\n")
    
    f.write("2. Cluster Sizes:\n")
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    for cluster, size in cluster_sizes.items():
        f.write(f"Cluster {cluster}: {size} users\n")
    
    f.write("\n3. Cluster Characteristics (Mean Values):\n")
    f.write(cluster_means.to_string())
    
    f.write("\n\n4. Cluster Profiles:\n")
    for cluster in range(optimal_k):
        f.write(f"\nCluster {cluster} Profile:\n")
        for feature in features_for_clustering:
            mean_val = cluster_means.loc[cluster, feature]
            f.write(f"{feature}: {mean_val:.2f}\n")

print("\nClustering Analysis complete! Additional files created:")
print("- 13_kmeans_optimal_k.png")
print("- 14_kmeans_clusters.png")
print("- 15_hierarchical_clustering.png")
print("- 16_cluster_characteristics.png")
print("- clustering_results.txt")
