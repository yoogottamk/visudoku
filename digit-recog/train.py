import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in).squeeze()
pickle_in.close()

m, n, p = X.shape

X = X.reshape(m, n*p)

pickle_in = open("y.pickle", "rb")
y = np.array(pickle.load(pickle_in))
pickle_in.close()

print("Loaded data")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Starting training")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Training ended")

pickle_out = open("knn.model", "wb")
pickle.dump(knn, pickle_out)
pickle_out.close()

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
