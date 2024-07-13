import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)
num_samples = 1000

# Features
word_freq_make = np.random.uniform(0, 1, num_samples)
word_freq_address = np.random.uniform(0, 1, num_samples)
word_freq_all = np.random.uniform(0, 1, num_samples)
word_freq_3d = np.random.uniform(0, 1, num_samples)
word_freq_our = np.random.uniform(0, 1, num_samples)
word_freq_over = np.random.uniform(0, 1, num_samples)
word_freq_remove = np.random.uniform(0, 1, num_samples)
word_freq_internet = np.random.uniform(0, 1, num_samples)
email_length = np.random.randint(50, 500, num_samples)

# Target variable (spam or not spam)
spam = (0.3 * word_freq_make + 0.2 * word_freq_address + 0.4 * word_freq_all - 
        0.5 * word_freq_3d + 0.1 * word_freq_our + 0.2 * word_freq_over + 
        0.3 * word_freq_remove - 0.4 * word_freq_internet + 
        0.01 * email_length + np.random.randn(num_samples) > 0.5).astype(int)

# Create a DataFrame
data = pd.DataFrame({
    'word_freq_make': word_freq_make,
    'word_freq_address': word_freq_address,
    'word_freq_all': word_freq_all,
    'word_freq_3d': word_freq_3d,
    'word_freq_our': word_freq_our,
    'word_freq_over': word_freq_over,
    'word_freq_remove': word_freq_remove,
    'word_freq_internet': word_freq_internet,
    'email_length': email_length,
    'spam': spam
})

# Select features and target variable
features = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
            'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
            'email_length']
target = 'spam'

X = data[features]
y = data[target]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
model = SVC(kernel='rbf', random_state=42)
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
plt.xticks(tick_marks, ['Not Spam', 'Spam'], rotation=45)
plt.yticks(tick_marks, ['Not Spam', 'Spam'])

thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
