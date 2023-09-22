import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_excel("/Users/batuhanoran/Desktop/BA_DATA/whole_data_sorted.xlsx")
x = data["Abstand"]
y = data["Vmag"]
y_filled = y.fillna(method='bfill')

# Normalize the input features
x_normalized = (x - x.mean()) / x.std()

# Define the degree of the polynomial
degree = 2

# Create the polynomial features
X_poly = np.column_stack([x_normalized**i for i in range(1, degree+1)])

# Add a column of ones for the intercept term
X = np.column_stack((np.ones(len(x_normalized)), X_poly))

# Define the learning rate and number of iterations
learning_rate = 0.01
num_iterations = 1000

# Initialize the parameters
theta = np.zeros(degree+1)

# Perform gradient descent
for _ in range(num_iterations):
    predictions = np.dot(X, theta)
    errors = predictions - y_filled.values
    gradient = np.dot(X.T, errors) / len(y_filled)
    theta -= learning_rate * gradient

# Retrieve the optimized parameters
intercept = theta[0]
coefficients = theta[1:]

# Create the regression equation
equation = f"Mmax = {intercept:.2f}"
for i, coefficient in enumerate(coefficients):
    equation += f" + {coefficient:.2f} * (Distance^{i+1})"
equation += f" / {x.std():.2f}"

# Printing the equation
print("Regression Equation:")
print(equation)

# Calculate the predictions using the optimized parameters
predictions = intercept + np.dot(X_poly, coefficients)

# Calculate the total sum of squares (TSS)
tss = np.sum((y_filled - y_filled.mean())**2)

# Calculate the residual sum of squares (RSS)
rss = np.sum((y_filled - predictions)**2)

# Calculate the R-squared value
r_squared = 1 - (rss / tss)

std_deviation_train = np.std(y_filled)
upper_bound = predictions + std_deviation_train
lower_bound = predictions - std_deviation_train

# Print the R-squared value
print("R-squared value:", r_squared)
print("Coefficients:", coefficients)
print("Intercept:", intercept)
print("Learning Rate:", learning_rate)

# Plotting the results
plt.figure()
plt.plot(x, y_filled, "k.")
plt.plot(x, predictions, color="red")
plt.plot(x, upper_bound, color='blue', linestyle='-')
plt.plot(x, lower_bound, color='blue', linestyle='-')
plt.xlabel("Distance (µm)")
plt.ylabel("Vmag")
plt.grid(True)
plt.show()
