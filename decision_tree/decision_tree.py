import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import plot_tree

# Generate synthetic data
np.random.seed(42)
num_samples = 1000

# Features
age = np.random.randint(18, 60, num_samples)
job_satisfaction = np.random.randint(1, 5, num_samples)
years_at_company = np.random.randint(1, 20, num_samples)
salary = np.random.randint(30000, 120000, num_samples)

# Target variable (attrition)
attrition = (0.05 * age - 0.1 * job_satisfaction + 0.03 * years_at_company + 0.0001 * salary + np.random.randn(num_samples) > 0).astype(int)

# Create a DataFrame
data = pd.DataFrame({
    'age': age,
    'job_satisfaction': job_satisfaction,
    'years_at_company': years_at_company,
    'salary': salary,
    'attrition': attrition
})

# Select features and target variable
features = ['age', 'job_satisfaction', 'years_at_company', 'salary']
target = 'attrition'

X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Plotting the decision tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=features, class_names=['Stay', 'Leave'], filled=True)
plt.title('Decision Tree for Employee Attrition Prediction')
plt.show()
