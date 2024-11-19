import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Set style
plt.style.use('default')

# Load the dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'user_behavior_dataset.csv')
df = pd.read_csv(csv_path)

# 1. Gender Distribution
plt.figure(figsize=(10, 6))
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.savefig('1_gender_distribution.png', bbox_inches='tight', dpi=300)
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
plt.savefig('2_device_models_distribution.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# 3. Apps vs Screen Time Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Number of Apps Installed', y='Screen On Time (hours/day)')
plt.title('Number of Apps vs Screen Time')
plt.xlabel('Number of Apps Installed')
plt.ylabel('Screen Time (hours/day)')
plt.savefig('3_apps_vs_screentime.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

correlation = df['Number of Apps Installed'].corr(df['Screen On Time (hours/day)'])
print(f"\nCorrelation between Apps Installed and Screen Time: {correlation:.2f}")

# 4. Screen Time by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Gender', y='Screen On Time (hours/day)')
plt.title('Screen Time Distribution by Gender')
plt.ylabel('Screen Time (hours/day)')
plt.savefig('4_screentime_by_gender.png', bbox_inches='tight', dpi=300)
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
plt.savefig('5_age_screentime_gender.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# 6. App Usage Time vs Screen Time
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='App Usage Time (min/day)', y='Screen On Time (hours/day)')
plt.title('App Usage Time vs Screen Time')
plt.xlabel('App Usage Time (minutes/day)')
plt.ylabel('Screen Time (hours/day)')
plt.savefig('6_app_usage_vs_screentime.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# 7. User Behavior Class Distribution
plt.figure(figsize=(10, 6))
behavior_counts = df['User Behavior Class'].value_counts()
sns.barplot(x=behavior_counts.index, y=behavior_counts.values)
plt.title('Distribution of User Behavior Classes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('7_user_behavior_distribution.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# 8. Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('8_age_distribution.png', bbox_inches='tight', dpi=300)
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
plt.savefig('9_correlation_matrix.png', bbox_inches='tight', dpi=300)
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
