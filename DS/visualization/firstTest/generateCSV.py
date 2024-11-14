# Import required libraries
import pandas as pd
import numpy as np
from numpy.random import choice
import random

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_students = 100

# Create student data
data = {
    'student_id': range(1001, 1001 + n_students),
    'age': np.random.randint(18, 30, n_students),
    'gender': choice(['M', 'F', 'Other'], n_students, p=[0.48, 0.48, 0.04]),
    'major': choice(['Computer Science', 'Engineering', 'Business', 'Biology', 'Psychology'], n_students),
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

# Add some correlation between variables
# Higher GPA tends to correlate with more study hours
df.loc[df['study_hours_per_week'] > 25, 'gpa'] += 0.3
df['gpa'] = df['gpa'].clip(0, 4.0)

# Higher stress levels tend to correlate with lower sleep hours
df.loc[df['stress_level'] > 7, 'sleep_hours_per_day'] -= 1
df['sleep_hours_per_day'] = df['sleep_hours_per_day'].clip(4, 10)

# Save to CSV
df.to_csv('student_data.csv', index=False)

# Display first few rows and basic information
print("First few rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.describe())