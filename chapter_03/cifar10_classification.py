from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
y_train = y_train.flatten()
y_test = y_test.flatten()

x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=0.1, random_state=42)
x_test, _, y_test, _ = train_test_split(x_test, y_test, train_size=0.1, random_state=42)

clf = svm.SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
train_pred = clf.predict(x_train)

print(classification_report(y_train, train_pred))
print(classification_report(y_test, y_pred))