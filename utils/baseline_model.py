import numpy as np
import pandas as pd
import datetime
from functools import partial
from scipy import stats

from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

from utils import vectorized_funs

#
#	In : 
#		Daily Changes of OPEN, HIGH, LOW, VOLUME
#
#	Out :
#		Returns after multiple timespans
#
#	Model:
#		2 Fully Connected Layers
#	
#	Feature Selection/Reduction :
#		None
#
#	Eval:
#		Accuracy on LOSS/GAIN prediction



def calc_features(raw_df, verbose=True):
	"""
	"""
	result_df = None

	result_df = pd.DataFrame()
	result_df["Date"] = raw_df["Date"]
	result_df["CHANGE_OPEN_1"] = ((raw_df["Open"] / raw_df["Open"].shift(1)) - 1)
	result_df["CHANGE_HIGH_1"] = ((raw_df["High"] / raw_df["High"].shift(1)) - 1)
	result_df["CHANGE_LOW_1"] = ((raw_df["Low"] / raw_df["Low"].shift(1)) - 1)
	result_df["CHANGE_VOLUME_1"] = ((raw_df["Volume"] / raw_df["Volume"].shift(1)) - 1)


	return result_df



def calc_labels(raw_df, timespan, verbose=True):
	"""
	"""
	result_df = None


	result_df = pd.DataFrame()
	result_df["Date"] = raw_df["Date"]

	for tp in timespan:
		for t in timespan[tp]:
			col_name = "RETURN_{}".format(t)
			result_df[col_name] = ((raw_df["Close"] / raw_df["Close"].shift(t)) - 1)
			result_df[col_name] = result_df[col_name].shift(-t)


	return result_df

def prepare_problemspace(features_df, labels_df, train_from, train_until, test_from, return_type="numpy"):
	"""
	"""
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

	if return_type == "numpy":
		X_train = X_train.values
		y_train = y_train.values
		X_test = X_test.values
		y_test = y_test.values

	return X_train, y_train, X_test, y_test

def create_model():
	"""
	"""

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

def fit(model, X_train, y_train, nb_epoch=1):
	"""
	"""

	model.fit(X_train, y_train, epochs=nb_epoch, batch_size=128, verbose=1)

	return model

def evaluate(model, X_test, y_test):
	"""
	"""
	_r = dict()

	y_pred = model.predict(X_test, verbose=0)
	gain_test = (y_test > 0.0) * 1.0
	gain_pred = (y_pred > 0.0) * 1.0

	_r["mse"] = mean_squared_error(y_true, y_pred)
	_r["r_squared"] = r2_score(y_test, y_pred, multioutput = "uniform_average")
	_r["accuracy"] = accuracy_score(gain_test, gain_pred)

	return _r
