import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4096, kernel_initializer='normal', activation="relu"))
	model.add(Dense(4096, kernel_initializer='normal', activation="relu"))
	# model.add(Dense(4096, kernel_initializer='normal', activation="relu"))
	# model.add(Dense(4096, kernel_initializer='normal', activation="relu"))
	# model.add(Dense(4096, kernel_initializer='normal', activation="relu"))
	
	#output
	#model.add(Dense(4, kernel_initializer='normal', activation="sigmoid"))
	model.add(Dense(4, kernel_initializer='normal'))
	# Compile model
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def baseline_train_test_split(features_df, labels_df, train_from, train_until, test_from):
	X_train = None
	y_train = None
	X_test = None
	y_test = None

	## Clear Infinity
	for col in features_df.columns.tolist():
		features_df.loc[np.isinf(features_df[col]), col] = 0.0
		features_df.loc[np.isnan(features_df[col]), col] = 0.0

	for col in labels_df.columns.tolist():
		labels_df.loc[np.isinf(labels_df[col]), col] = 0.0
		labels_df.loc[np.isnan(labels_df[col]), col] = 0.0

	X_train = features_df.loc[train_from:train_until, :]
	y_train = labels_df.loc[train_from:train_until, :]
	X_test = features_df.loc[test_from:, :]
	y_test = labels_df.loc[test_from:, :]

	X_train = X_train.values
	y_train = y_train.values
	X_test = X_test.values
	y_test = y_test.values

	return X_train, y_train, X_test, y_test


def baseline_fit_and_eval(model, X_train, y_train, X_test, y_test):
	_r = None

	model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

	y_pred = model.predict(X_test, verbose=0)

	_r = accuracy_score((y_test > 0.0) * 1.0, (y_pred > 0.0) * 1.0)

	return _r


