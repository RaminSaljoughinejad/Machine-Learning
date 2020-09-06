import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

data = datasets.load_breast_cancer()

x = data.data
y = data.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ["Malignant", "Benign"]

model = svm.SVC(kernel="linear")

model.fit(x_train, y_train)

predictions = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, predictions)

print("SVM: ", accuracy)

model1 = KNeighborsClassifier(n_neighbors=9)
model1.fit(x_train, y_train)

prediction1 = model1.predict(x_test)
accuracy1 = metrics.accuracy_score(y_test, prediction1)

print("KNN: ", accuracy1)