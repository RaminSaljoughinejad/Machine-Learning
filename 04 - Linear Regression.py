import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import pickle

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    accuracy = linear.score(x_test,y_test)

    if accuracy > best:
        best = accuracy
        with open("studentGradeModel.pickle", "wb") as modelFile:
            pickle.dump(linear, modelFile)

print("Highest Accuracy: ", best)
print("Best Model Saved.")