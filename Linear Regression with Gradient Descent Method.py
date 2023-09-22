import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_excel("/Users/batuhanoran/Desktop/BA_DATA/whole_data_sorted.xlsx")
x = data["Abstand"]
y = data["Mr"]
y_filled = y.fillna(method='bfill')

x_train, x_test, y_train, y_test = train_test_split(x, y_filled, test_size=0.2, random_state=42)

# Normalize the input features for training set
x_train_normalized = (x_train - x_train.mean()) / x_train.std()

# Add a column of ones for the intercept term for training set
X_train = np.column_stack((np.ones(len(x_train_normalized)), x_train_normalized))

# Define the learning rate and number of iterations
learning_rate = 0.01
num_iterations = 1000

# Initialize the parameters
theta = np.zeros(2)

# Perform gradient descent for training set
for _ in range(num_iterations):
    predictions = np.dot(X_train, theta)
    errors = predictions - y_train.values
    gradient = np.dot(X_train.T, errors) / len(y_train)
    theta -= learning_rate * gradient

# Retrieve the optimized parameters for training set
intercept = theta[0]
coefficients = theta[1:]

# Calculate the predictions for training set using the optimized parameters
predictions_train = intercept + coefficients * ((x_train - x_train.mean()) / x_train.std())

# Calculate the total sum of squares (TSS) for training set
tss_train = np.sum((y_train - y_train.mean())**2)

# Calculate the residual sum of squares (RSS) for training set
rss_train = np.sum((y_train - predictions_train)**2)

# Calculate the R-squared value for training set
r_squared_train = 1 - (rss_train / tss_train)
mse_train = np.mean((y_train - predictions_train)**2)
# Print the R-squared value for training set
print("MSE for training set:", mse_train)
print("R-squared value for training set:", r_squared_train)

# Calculate the standard deviation for training set
std_deviation_train = np.std(y_train)

# Calculate the upper and lower bounds for shading
upper_bound = predictions_train + std_deviation_train
lower_bound = predictions_train - std_deviation_train

# Plotting the results
plt.figure()
plt.plot(x_train, y_train, "k.")
plt.plot(x_train, intercept + coefficients * x_train_normalized, color="red")
plt.plot(x_train, upper_bound, color='blue', linestyle='-')
plt.plot(x_train, lower_bound, color='blue', linestyle='-')
plt.xlabel("Distance (µm)")
plt.ylabel("Mr")
plt.legend()
plt.grid(True)
plt.show()
