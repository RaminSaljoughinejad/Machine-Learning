import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pickle


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

best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(x_train,y_train)

    accuracy = model.score(x_test, y_test)
    if accuracy > best:
        best = accuracy
        with open("carModel.pickle", "wb") as modelFile:
            pickle.dump(model, modelFile)
print("Best Accuracy: ",best)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

print("\n\n#################################")
for i in range(len(predicted)):
    print(f"Data #{i+1}")
    print("Model Prediction: ", names[predicted[i]])
    print("Input Data: ", x_test[i])
    print("Real Label: ", names[y_test[i]])
    print("#################################")