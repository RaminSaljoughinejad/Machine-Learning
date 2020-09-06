import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

data = datasets.load_breast_cancer()

x = data.data
y = data.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ["Malignant", "Benign"]

model = svm.SVC(kernel="linear")  # supper vector classifier --- poly can get a second input , degree(2) ---- c = soft margin -- c=2 soft margin!

model.fit(x_train, y_train)

predictions = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, predictions)

print(accuracy)