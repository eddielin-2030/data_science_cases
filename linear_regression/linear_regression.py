import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
num_samples = 100

# Features
num_bedrooms = np.random.randint(1, 6, num_samples)
size_sqft = np.random.randint(500, 3500, num_samples)
location = np.random.randint(1, 10, num_samples)
age = np.random.randint(1, 50, num_samples)

# Target variable (price)
price = (num_bedrooms * 50000) + (size_sqft * 200) + (location * 10000) - (age * 500) + np.random.randint(20000, 50000, num_samples)

# Create a DataFrame
data = pd.DataFrame({
    'num_bedrooms': num_bedrooms,
    'size_sqft': size_sqft,
    'location': location,
    'age': age,
    'price': price
})

# Select features and target variable
features = ['num_bedrooms', 'size_sqft', 'location', 'age']
target = 'price'

X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plotting the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()
