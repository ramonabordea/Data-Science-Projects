# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_students = 100

# Create student data
data = {
    'student_id': range(1001, 1001 + n_students),
    'age': np.random.randint(18, 30, n_students),
    'gender': np.random.choice(['M', 'F', 'Other'], n_students, p=[0.48, 0.48, 0.04]),
    'major': np.random.choice(['Computer Science', 'Engineering', 'Business', 'Biology', 'Psychology'], n_students),
    'gpa': np.round(np.random.normal(3.2, 0.5, n_students).clip(0, 4.0), 2),
    'study_hours_per_week': np.random.randint(5, 40, n_students),
    'sleep_hours_per_day': np.round(np.random.normal(7, 1, n_students).clip(4, 10), 1),
    'stress_level': np.random.randint(1, 11, n_students),  # Scale 1-10
    'extracurricular_activities': np.random.randint(0, 5, n_students),
    'commute_time_minutes': np.random.randint(10, 90, n_students)
}

# Create attendance data (percentage)
data['attendance_rate'] = np.round(np.random.uniform(60, 100, n_students), 1)

# Create satisfaction score
data['satisfaction_score'] = np.random.randint(1, 6, n_students)  # Scale 1-5

# Convert to DataFrame
df = pd.DataFrame(data)

# Add some correlations
df.loc[df['study_hours_per_week'] > 25, 'gpa'] += 0.3
df['gpa'] = df['gpa'].clip(0, 4.0)
df.loc[df['stress_level'] > 7, 'sleep_hours_per_day'] -= 1
df['sleep_hours_per_day'] = df['sleep_hours_per_day'].clip(4, 10)

# Save to CSV
df.to_csv('student_data.csv', index=False)

# Basic EDA
print("\nDataset Info:")
print(df.info())
print("\nNumerical Variables Summary:")
print(df.describe())

# Create visualizations
# Set the style for all plots
sns.set_style("whitegrid")

# 1. Distribution plots for numerical variables
numerical_cols = ['age', 'gpa', 'study_hours_per_week', 'sleep_hours_per_day', 
                 'stress_level', 'attendance_rate', 'satisfaction_score']

for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'dist_{col}.png')
    plt.close()

# 2. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# 3. Box plots for numerical variables by gender
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='gender', y=col)
    plt.title(f'{col} by Gender')
    plt.savefig(f'boxplot_{col}_by_gender.png')
    plt.close()

# 4. Count plots for categorical variables
categorical_cols = ['gender', 'major']
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=45)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(f'count_{col}.png')
    plt.close()

# 5. Scatter plot: GPA vs Study Hours
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='study_hours_per_week', y='gpa', hue='gender')
plt.title('GPA vs Study Hours per Week')
plt.savefig('scatter_gpa_study_hours.png')
plt.close()

print("\nEDA completed! Visualizations have been saved as PNG files.")