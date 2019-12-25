import pickle

model_pickle = open('knn.model', 'rb')
model = pickle.load(model_pickle)
model_pickle.close()

print(model.predict([[0] * 900]))
