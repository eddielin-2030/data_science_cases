import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
num_samples = 1000

# Features
credit_score = np.random.randint(300, 850, num_samples)
income = np.random.randint(20000, 150000, num_samples)
loan_amount = np.random.randint(1000, 50000, num_samples)
employment_history = np.random.randint(0, 20, num_samples)

# Target variable (default or not)
default = (0.3 * (credit_score < 600) + 0.2 * (income < 40000) + 0.4 * (loan_amount > 20000) + 
           0.1 * (employment_history < 2) + np.random.randn(num_samples) > 0.5).astype(int)

# Create a DataFrame
data = pd.DataFrame({
    'credit_score': credit_score,
    'income': income,
    'loan_amount': loan_amount,
    'employment_history': employment_history,
    'default': default
})

# Select features and target variable
features = ['credit_score', 'income', 'loan_amount', 'employment_history']
target = 'default'

X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

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
plt.xticks(tick_marks, ['No Default', 'Default'], rotation=45)
plt.yticks(tick_marks, ['No Default', 'Default'])

thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
