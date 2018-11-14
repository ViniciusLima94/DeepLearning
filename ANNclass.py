# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ANNclass import ANN

def score(y, y_pred):
	'''
		Scores accuracy of ANN.
		Inputs:
		y: Real value to be predict
		y_pred: Predictions made be the ANN.
		Outputs:
		score: Pontuation score (%) of the ANN.
	'''
	score = 0
	for i in range(len(y)):
		if y[i] == y_pred[i]:
			score += 1
	return float(score) / len(y) * 100

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Reshaping y array
y_train = y_train.reshape((X_train.shape[0],1))
y_test  = y_test.reshape((X_test.shape[0],1))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = ANN(layers=(6,6,1), X=X_train, y=y_train, batch_size = 10, epochs=100, alpha=20, seed=546)
ann.fit()
y_pred = ann.predict(X_test)
# Binarizing y_pred
y_pred = (y_pred>0.5).astype(int)

acc = score(y_test, y_pred)
print 'ANN accuracy is ' + str(acc) + '%.'

#ann.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
