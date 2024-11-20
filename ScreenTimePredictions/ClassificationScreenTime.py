import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import os

# Create directories if they don't exist
current_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_dir, 'plots')
analysis_dir = os.path.join(current_dir, 'analysis')

for directory in [plots_dir, analysis_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

print("Starting Classification Analysis...")
print("=" * 50)

# Load the dataset

csv_path = os.path.join(current_dir, 'user_behavior_dataset.csv')
df = pd.read_csv(csv_path)


# Prepare data for classification
X = df.drop(['User ID', 'Device Model', 'Operating System', 'User Behavior Class'], axis=1)
y = df['User Behavior Class']

# Encode categorical variables
le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

# Train and evaluate models
results = {}
print("\nModel Performance:")
print("-" * 50)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = model.score(X_test_scaled, y_test)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'predictions': y_pred
    }
    
    print(f"{name} Accuracy: {accuracy:.4f}")

# Get best model
best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
best_model = models[best_model_name]
best_predictions = results[best_model_name]['predictions']

print(f"\nBest Model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")

# Plot confusion matrix for best model
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Feature importance for Random Forest
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Classification')
plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Model comparison plot
plt.figure(figsize=(10, 6))
accuracies = {name: result['accuracy'] for name, result in results.items()}
plt.bar(accuracies.keys(), accuracies.values())
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Cross-validation scores
print("\nCross-validation scores:")
print("-" * 50)
cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_scores[name] = (scores.mean(), scores.std())
    print(f"{name} CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Save classification results
with open(os.path.join(analysis_dir, 'classification_results.txt'), 'w') as f:
    f.write("Classification Analysis Results\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. Model Accuracies:\n")
    for name, result in results.items():
        f.write(f"{name}: {result['accuracy']:.4f}\n")
    
    f.write(f"\n2. Best Model: {best_model_name}\n")
    f.write(f"Accuracy: {results[best_model_name]['accuracy']:.4f}\n")
    
    f.write("\n3. Detailed Classification Report for Best Model:\n")
    f.write(classification_report(y_test, best_predictions))
    
    f.write("\n4. Feature Importance (Random Forest):\n")
    for idx, row in feature_importance.iterrows():
        f.write(f"{row['feature']}: {row['importance']:.4f}\n")
    
    f.write("\n5. Cross-validation Scores:\n")
    for name, (mean_score, std_score) in cv_scores.items():
        f.write(f"{name}: {mean_score:.4f} (+/- {std_score * 2:.4f})\n")

# Save model comparison results
with open(os.path.join(analysis_dir, 'model_comparison.txt'), 'w') as f:
    f.write("Model Comparison Results\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. Accuracy Ranking:\n")
    sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    for i, (name, acc) in enumerate(sorted_models, 1):
        f.write(f"{i}. {name}: {acc:.4f}\n")
    
    f.write("\n2. Cross-validation Performance:\n")
    for name, (mean_score, std_score) in cv_scores.items():
        f.write(f"{name}:\n")
        f.write(f"  Mean CV Score: {mean_score:.4f}\n")
        f.write(f"  Standard Deviation: {std_score:.4f}\n")

print("\nClassification Analysis complete!")
print(f"Plots saved in: {plots_dir}")
print("- confusion_matrix.png")
print("- feature_importance.png")
print("- model_comparison.png")
print(f"\nAnalysis saved in: {analysis_dir}")
print("- classification_results.txt")
print("- model_comparison.txt") 