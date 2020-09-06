import sklearn
from sklearn import datasets
from sklearn import svm

data = datasets.load_breast_cancer()

print(len(data.feature_names), "---->", data.feature_names, "\n")
print(len(data.target_names), "---->", data.target_names)

#############

x = data.data
y = data.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(x_train)
print(y_train)

#############

classes = ["Malignant", "Benign"]