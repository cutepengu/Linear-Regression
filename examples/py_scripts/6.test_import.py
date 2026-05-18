import exporter
import pickle
import numpy as np
import os
#Create a folder for the output files
if not os.path.exists("../output"):
    os.makedirs("../output")

exporter.save_model()

loaded_model = pickle.load(open('../output/my_saved_model.sav', 'rb'))
predict = np.array([4]).reshape(1, -1)
result = loaded_model.predict(predict)
print(result[0])

