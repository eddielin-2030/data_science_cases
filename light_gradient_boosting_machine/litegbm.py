import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
num_samples = 1000

# Features
transaction_time = np.random.randint(0, 86400, num_samples)  # Seconds in a day
transaction_amount = np.random.uniform(1, 1000, num_samples)
merchant_category = np.random.choice(['grocery', 'electronics', 'clothing', 'restaurants', 'travel'], num_samples)
merchant_category_num = pd.factorize(merchant_category)[0]

# Target variable (fraud or not)
fraud = (0.3 * (transaction_amount > 500) + 0.2 * (merchant_category_num == 1) + 0.4 * (transaction_time < 43200) + 
         0.1 * (merchant_category_num == 3) + np.random.randn(num_samples) > 0.5).astype(int)

# Create a DataFrame
data = pd.DataFrame({
    'transaction_time': transaction_time,
    'transaction_amount': transaction_amount,
    'merchant_category': merchant_category_num,
    'fraud': fraud
})

# Select features and target variable
features = ['transaction_time', 'transaction_amount', 'merchant_category']
target = 'fraud'

X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the LightGBM model
train_data = lgb.Dataset(X_train, label=y_train)
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}
model = lgb.train(params, train_data, num_boost_round=100)

# Make predictions on the test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# Print evaluation metrics
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Fraud', 'Fraud'], rotation=45)
plt.yticks(tick_marks, ['No Fraud', 'Fraud'])

thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
