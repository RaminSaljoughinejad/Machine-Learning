import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing


data = pd.read_csv("car.data")
#print(data.head())

myPreprocessor = preprocessing.LabelEncoder()
buying = myPreprocessor.fit_transform(list(data["buying"]))
maint = myPreprocessor.fit_transform(list(data["maint"]))
door = myPreprocessor.fit_transform(list(data["door"]))
persons = myPreprocessor.fit_transform(list(data["persons"]))
lug_boot = myPreprocessor.fit_transform(list(data["lug_boot"]))
safety = myPreprocessor.fit_transform(list(data["safety"]))
clas = myPreprocessor.fit_transform(list(data["class"]))

predit = "class"

x = list(zip(buying, maint, door, lug_boot, safety))
y = list(clas)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

print(x_train)