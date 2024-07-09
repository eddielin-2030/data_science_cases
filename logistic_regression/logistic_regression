import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Generate synthetic data
np.random.seed(42)
num_samples = 1000

# Features
tenure = np.random.randint(1, 60, num_samples)
monthly_charges = np.random.uniform(20, 100, num_samples)
contract_type = np.random.choice([0, 1], num_samples)  # 0: Month-to-month, 1: One year or Two years
support_calls = np.random.randint(0, 10, num_samples)

# Target variable (churn)
churn = (0.1 * tenure + 0.05 * monthly_charges - 0.2 * contract_type + 0.1 * support_calls + np.random.randn(num_samples) > 0.5).astype(int)

# Create a DataFrame
data = pd.DataFrame({
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'contract_type': contract_type,
    'support_calls': support_calls,
    'churn': churn
})

# Select features and target variable
features = ['tenure', 'monthly_charges', 'contract_type', 'support_calls']
target = 'churn'

X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression()
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

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Not Churn', 'Churn'], rotation=45)
plt.yticks(tick_marks, ['Not Churn', 'Churn'])

thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
