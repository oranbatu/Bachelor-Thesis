import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_excel("/Users/batuhanoran/Desktop/BA_DATA/whole_data_sorted.xlsx")
x = data["Abstand"]
y = data["Re4"]
y_filled = y.fillna(method='bfill')

x_train, x_test, y_train, y_test = train_test_split(x, y_filled, test_size=0.2, random_state=42)

# Preprocess the data by adding a column of ones to represent the intercept term
X = np.column_stack((np.ones(len(x_train)), x_train))

# Compute the coefficients using the OLS formula
coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_train)

# Extract the intercept and slope from the coefficients
intercept = coefficients[0]
slope = coefficients[1]

std_deviation_train = np.std(y_train)

# Create the regression equation
equation = f"Mmax = {intercept:.2f} + {slope:.2f} * Distance + {std_deviation_train:.2f}"

# Print the equation
print("Regression Equation:")
print(equation)
print(intercept)
print(coefficients)

# Generate predicted values using the computed coefficients
y_predicted = intercept + slope * x_train

# Calculate R-squared
y_mean = np.mean(y_train)
ss_total = np.sum((y_train - y_mean)**2)
ss_residual = np.sum((y_train - y_predicted)**2)
r_squared = 1 - (ss_residual / ss_total)
print("R-squared for Training set:", r_squared)

# Calculate Mean Squared Error (MSE)
mse = np.mean((y_train - y_predicted)**2)
mse_train_formatted = np.format_float_positional(mse, precision=9)
print("MSE for training set:", mse_train_formatted)

# Generate predicted values for the test set using the computed coefficients
y_predicted_test = intercept + slope * x_test

# Calculate R-squared for the test set
ss_total_test = np.sum((y_test - y_mean)**2)
ss_residual_test = np.sum((y_test - y_predicted_test)**2)
r_squared_test = 1 - (ss_residual_test / ss_total_test)
print("R-squared for Test set:", r_squared_test)

# Calculate Mean Squared Error (MSE) for the test set
mse_test = np.mean((y_test - y_predicted_test)**2)
mse_test_formatted = np.format_float_positional(mse_test, precision=9)
print("MSE for Test set:", mse_test_formatted)

print(intercept)
print(coefficients)

upper_bound = y_predicted + std_deviation_train
lower_bound = y_predicted - std_deviation_train

# Plot the results
plt.figure()
plt.plot(x_train, y_train, "k.")
plt.plot(x_train, y_predicted, color="red")
plt.errorbar(x_train, y_train, yerr=std_deviation_train, fmt=".", color="black", ecolor="orange", elinewidth=0.7, capsize=3, alpha=1, label="Standard Deviation")
plt.xlabel("Distance (µm)")
plt.ylabel("Re4")
plt.grid(True)
plt.show()
