import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_excel("/Users/batuhanoran/Desktop/BA_DATA/whole_data_sorted.xlsx")
x = data["Abstand"]
y = data["Vmag"]
y_filled = y.fillna(method='bfill')

X_train, X_test, y_train, y_test = train_test_split(x, y_filled, test_size=0.2, random_state=42)
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

model = DecisionTreeRegressor(max_depth=10)

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
r2_cv = cv_scores
r2_cv_mean = r2_cv.mean()

# Fit the model on the training data
model.fit(X_train, y_train)

# Evaluate on the test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = model.score(X_test, y_test)

# Format the R-squared values
r2_cv_formatted = [f"{score:.6f}" for score in r2_cv]
r2_cv_mean_formatted = "{:.6f}".format(r2_cv_mean)
r2_formatted = "{:.6f}".format(r2)

print("CV R-squared scores:", r2_cv_formatted)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared:", r2_formatted)

plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=["Abstand"], filled=True)
plt.show()
