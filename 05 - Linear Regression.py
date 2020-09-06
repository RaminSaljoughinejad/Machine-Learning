import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

# Data
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Loading Model
pickle_in = open("studentGradeModel.pickle", "rb")
linear = pickle.load(pickle_in)


style.use("ggplot")
plt.scatter(data["G1"],data["G3"], label="G1")
#plt.scatter(data["G2"],data["G3"], label="G2")
#plt.scatter(data["studytime"],data["G3"], label="studytime")
#plt.scatter(data["failures"],data["G3"], label="Failures")
#plt.scatter(data["absences"],data["G3"], label="Absences")
plt.xlabel("P")
plt.ylabel("Final Grade")
plt.legend()
plt.show()