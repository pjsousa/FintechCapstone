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


feature_cols = ['timewindow_return_1_Open', 'timewindow_return_1_High',
				'timewindow_return_1_Low',
				'timewindow_return_1_Volume']

labels_cols = ['timewindow_return_1_Close',
		'timewindow_return_5_Close', 
		'timewindow_return_90_Close',
		'timewindow_return_40_Close']

labels_cols = ['timewindow_return_1_Close']

def baseline_binary_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(32, kernel_initializer='normal', activation="relu"))
	model.add(Dense(32, kernel_initializer='normal', activation="relu"))
	model.add(Dense(1, kernel_initializer='normal', activation="sigmoid"))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
	return model

def baseline_train_test_split(itr_df, train_from, train_until, test_from):
	X_train = None
	y_train = None
	X_test = None
	y_test = None

	## Clear Infinity
	for col in feature_cols:
		itr_df.loc[np.isinf(itr_df[col]), col] = 0.0

	for col in labels_cols:
		itr_df.loc[np.isinf(itr_df[col]), col] = 0.0

	X_train = itr_df.loc[train_from:train_until, feature_cols]
	y_train = itr_df.loc[train_from:train_until, labels_cols]
	X_test = itr_df.loc[test_from:, feature_cols]
	y_test = itr_df.loc[test_from:, labels_cols]

	y_train = (y_train > 0.0) * 1.0
	y_test = (y_test > 0.0) * 1.0

	X_train = X_train.values
	y_train = y_train.values
	X_test = X_test.values
	y_test = y_test.values

	return X_train, y_train, X_test, y_test


def baseline_fit_and_eval(model, X_train, y_train, X_test, y_test):
	_r = None

	model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)

	_eval = model.evaluate(X_test, y_test, verbose=0)

	try:
		_r = _eval[1]
	except:
		_r = np.nan

	return _r

