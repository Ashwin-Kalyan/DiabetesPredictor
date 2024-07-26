import warnings
warnings.simplefilter("ignore", UserWarning)

import pandas as pd
import numpy as np
import csv
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

data = pd.read_csv("diabetes.csv") # data provided by Kaggle
print(data)
x = data.drop(columns='Outcome', axis=1)
y = data['Outcome'] 

scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)

x = standardized_data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# accuracy score for model
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train) # 78.66% - pretty good

x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test) # 77.27% - pretty good

# prediction system
with open("diabetes.csv") as f:
    input_data=[tuple(line[0:8]) for line in csv.reader(f)]
    input_data.pop(0) # redundant tuple of column names
    for line in input_data:
        in_data_np_arr = np.asarray(line)
        np_arr_reshaped = in_data_np_arr.reshape(1, -1)
        std_in_data = scaler.transform(np_arr_reshaped)
        prediction = classifier.predict(std_in_data) 
        print(prediction)