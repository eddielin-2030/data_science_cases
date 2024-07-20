import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
num_samples = 1000

# Features
contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples)
monthly_charges = np.random.uniform(20, 120, num_samples)
tenure = np.random.randint(0, 72, num_samples)
total_charges = monthly_charges * tenure

# Convert contract_type to numerical values
contract_type_num = np.where(contract_type == 'Month-to-month', 0, np.where(contract_type == 'One year', 1, 2))

# Target variable (churn or not)
churn = (0.3 * (contract_type_num == 0) + 0.2 * (monthly_charges > 80) + 0.4 * (tenure < 12) + 
         0.1 * (total_charges < 1000) + np.random.randn(num_samples) > 0.5).astype(int)

# Create a DataFrame
data = pd.DataFrame({
    'contract_type': contract_type_num,
    'monthly_charges': monthly_charges,
    'tenure': tenure,
    'total_charges': total_charges,
    'churn': churn
})

# Select features and target variable
features = ['contract_type', 'monthly_charges', 'tenure', 'total_charges']
target = 'churn'

X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Gradient Boosting model
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
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
plt.xticks(tick_marks, ['No Churn', 'Churn'], rotation=45)
plt.yticks(tick_marks, ['No Churn', 'Churn'])

thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
