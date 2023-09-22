import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime


dfs = []  # List to store individual dataframes

for file_path in file_paths:
    df = pd.read_excel(file_path)  # Read each Excel file
    dfs.append(df)  # Append dataframe to the list

combined_df = pd.concat(dfs)  # Concatenate all dataframes into one

x = data["Abstand"]
y = data["Mmax"]
y_filled = y.fillna(method='bfill')

x_train, x_test, y_train, y_test = train_test_split(x, y_filled, test_size=0.2, random_state=42)

model = SVR(kernel='linear', C=1.0, epsilon=0.16)  # Create SVR model with linear kernel
model.fit(x_train.values.reshape(-1, 1), y_train)

# Reset indices of x_train and y_train
x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

y_train_predictions = model.predict(x_train.values.reshape(-1, 1))
mse_train = mean_squared_error(y_train, y_train_predictions)
print("MSE for training set: {:.9f}".format(mse_train))
print("R-squared for training set", r2_score(y_train, y_train_predictions))

y_test_predictions = model.predict(x_test.values.reshape(-1, 1))
mse_test = mean_squared_error(y_test, y_test_predictions)
print("MSE for test set: {:.9f}".format(mse_test))
print("r-squared for test set:", r2_score(y_test, y_test_predictions))

support_vectors = model.support_vectors_
support_vector_indices = model.support_

# Obtain equation of the line, coefficients, and intercept
slope = model.coef_[0]
intercept = model.intercept_

print("Equation of the line: Mmax =", slope, "* Abstand +", intercept)
print("Coefficients:", model.coef_)
print("Intercept:", intercept)

# Print current time
current_time = datetime.datetime.now().strftime("%H:%M:%S")
print("Current time:", current_time)

# Plotting
plt.figure()
plt.title("Regression")
plt.plot(x_train, y_train, "k.")
plt.plot(x_train, model.predict(x_train.values.reshape(-1, 1)), color="red", label="SVR Prediction")
plt.scatter(x_train[support_vector_indices], y_train[support_vector_indices], s=100, facecolors='none', edgecolors='blue', label="Support Vectors")
plt.xlabel("Distance (µm)")
plt.ylabel("Mmax")
plt.legend()
plt.grid(True)
plt.show()
