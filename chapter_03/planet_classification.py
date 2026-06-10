from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas

csv = pandas.read_csv(r'nasa.csv', encoding='unicode_escape')
csv = csv.drop(
    ['Neo Reference ID', 'Name', 'Close Approach Date',
     'Orbit Determination Date', 'Orbiting Body', 'Equinox'],
    axis=1
)

data = csv.values
x_data, y_data = data[:, :-1], data[:, -1].astype(int)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.20, random_state=42
)

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

clf = svm.SVC(kernel='linear', C=1.0, random_state=42, max_iter=100000)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
train_pred = clf.predict(x_train)

print(classification_report(y_train, train_pred))
print(classification_report(y_test, y_pred))