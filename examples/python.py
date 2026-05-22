# Import frameworks
import numpy as np
import pandas as pd
import sqlite3 as sql
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LinearRegression

# Create the data for feature and the target
training_data = pd.read_csv("data/3.course_specifications_data.csv", delimiter=',')
x = np.array(training_data.iloc[:,1]).reshape(-1, 1)
y = np.array(training_data.iloc[:,0])

m = len(x)
print(f"Number of training examples is: {m}")
table = pd.DataFrame({
    training_data.columns[0]: x.flatten(),  # Flatten x for easy display
    training_data.columns[1]: y
})
print(table.head())

# Plot the data points
plt.scatter(x, y, marker='x', c='r')
# Set the title
plt.title("NESA Course Specifications Data")
# Set the y-axis label
plt.ylabel(f'Training {training_data.columns[0]}')
# Set the x-axis label
plt.xlabel(f'Training {training_data.columns[1]}')
plt.show()

# Create the model
my_model = LinearRegression()

# Fit the model to the data
my_model.fit(x, y)

y_pred = my_model.predict(x)
plt.plot(x, y_pred)
plt.scatter(x, y, marker='x', c='r')
plt.title("NESA Course Specifications Data")
plt.ylabel(f'Training {training_data.columns[0]}')
plt.xlabel(f'Training {training_data.columns[1]}')
plt.show()

predict = np.array([[4]])
y_prediction = my_model.predict(predict)

y_pred = my_model.predict(x)
plt.plot(x, y_pred)
plt.scatter(x, y, marker='x', c='r')
plt.scatter(predict, y_prediction, marker='D', c='r', zorder=10, s=100)
plt.text(y_prediction, predict, f"Target {y_prediction[0]} is prediction from {predict[0,0]} input")
plt.title("NESA Course Specifications Data")
plt.ylabel(f'Training {training_data.columns[0]}')
plt.xlabel(f'Training {training_data.columns[1]}')
plt.show()