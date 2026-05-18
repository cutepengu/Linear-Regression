import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LinearRegression
import pickle
import os
#Create a folder for the output files
if not os.path.exists("../output"):
    os.makedirs("../output")


def save_model ():
    training_data = pd.read_csv('../data/3.course_specifications_data.csv', delimiter=',')
    x = np.array(training_data.iloc[:,1]).reshape(-1, 1)
    y = np.array(training_data.iloc[:,0])# Create the model
    my_model = LinearRegression()
    my_model.fit(x, y)
    filename = '../output/my_saved_model.sav'
    pickle.dump(my_model, open(filename, 'wb'))
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    save_model()