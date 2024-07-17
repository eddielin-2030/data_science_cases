import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
num_samples = 1000

# Example movie reviews (synthetic data with some noise)
reviews = [
    "I loved the movie, it was fantastic and thrilling!",
    "The movie was awful, I hated it and it was boring.",
    "Great movie with excellent plot and characters.",
    "Terrible movie, not worth the time.",
    "An amazing experience, would watch again!",
    "Worst movie ever, it was dull and uninteresting.",
    "It was an okay movie, had some good moments.",
    "Not my favorite, but had some decent scenes.",
    "Pretty good movie, enjoyed most parts.",
    "Bad movie, wouldn't recommend."
] * (num_samples // 10)

# Introduce some noise in labels
labels = ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * (num_samples // 10)) + np.random.randint(0, 2, num_samples // 5).tolist()

# Create a DataFrame
data = pd.DataFrame({
    'review': reviews,
    'label': labels[:num_samples]  # Ensure the length matches num_samples
})

# Select features and target variable
X = data['review']
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Create and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vec)

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
plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
plt.yticks(tick_marks, ['Negative', 'Positive'])

thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

