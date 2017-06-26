import itertools
import math
import re
import glob
import os

import numpy as np
import pandas as pd
import datetime
from functools import partial
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from utils import paths_helper as paths
from utils import vectorized_funs
from utils import kerasutil as kutils
from utils import scenarioa

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.metrics import binary_accuracy
from keras.wrappers.scikit_learn import KerasRegressor

TINY_FLOAT = 1e-10

#
#	in : 
#		multiple technical indicators:
#			simple moving average for 5, 30, 60 and 200 periods
#			bollinger bands for 5, 30, 60 and 200 periods
#			macd for 26/12 period
#			rsi for 14, 21, 60 periods
#			stochastic oscilator for 14 periods
#			adx
#			aroon indicator for 20 periods
#			chaikin money flow for 21 periods
#			on balance volume indicator
#
#	out :
#		returns after multiple timespans
#
#	feature selection/reduction :
#		none
#
#	model:
#		????
#
#	eval:
#		accuracy on loss/gain prediction
#
#
#       https://www.aaai.org/ocs/index.php/WS/AAAIW15/paper/viewFile/10179/10251
#

def accuracy_gainloss(y_true, y_pred):
    gain_test = K.cast(K.greater(y_true, K.constant(0.5)), K.floatx())
    gain_pred = K.cast(K.greater(y_pred, K.constant(0.5)), K.floatx())

    return binary_accuracy(gain_test, gain_pred)


def r2_regression(y_true, y_pred):
	numerator = K.square(y_true - y_pred)
	denominator = K.sum(K.square(y_true - K.mean(y_true)))

	return K.constant(1.0) - (numerator / denominator)



def markov_transition_matrix(serie, n_states, min_val, max_val):
	min_val = serie.min() if min_val is None else min_val
	max_val = serie.max() if max_val is None else max_val

	## Digitize the column and create the "next state" by shifting it
	states = pd.DataFrame()
	states["state_in"] = np.digitize(serie, np.linspace(min_val, max_val, n_states, False))
	states["state_out"] = states["state_in"].shift(-1).values.astype(np.int)
	states["pair_count"] = 1
	states = states.iloc[:-1] ## because of the shift, the last row will have a Nan converted to int, which is garbage

	## create the total count of obervations in each state
	states_count = states.loc[:,["state_in", "state_out"]].groupby(["state_in"]).count()

	## groupby to get the count of state_in/state_out pairs
	states = states.groupby(["state_in", "state_out"]).count()

	## Pivot it to create our
	states.reset_index(level=["state_in", "state_out"], inplace=True)
	states = states.pivot("state_in", "state_out").fillna(0)

	## get the frequency 
	#states = states / states_count.values.flatten()
	states = states.divide(states_count.values.flatten(), axis=0)

	## drop the dummy variable just so our output look nicer
	states.columns = states.columns.droplevel(0)


	## we might need to force the indexes if some states don't really happen
	_full_state = np.arange(0,n_states) + 1

	return pd.DataFrame(states.T, index=_full_state, columns=_full_state).fillna(0)


def markov_transition_field(serie, n_states, min_val, max_val):
	min_val = serie.min() if min_val is None else min_val
	max_val = serie.max() if max_val is None else max_val

	W = markov_transition_matrix(serie, n_states, min_val, max_val)

	## this will be the "state_in"
	i = pd.DataFrame(np.digitize(serie, np.linspace(min_val, max_val, n_states, False))).T
	i = i.append([i] * (serie.shape[0] - 1), ignore_index=True)

	# this will be the "state_out"
	j = i.T

	## stack i and j depth-wise (create pairs that go from the i-th state to the j-th state)
	M = np.dstack((i, j))

	## find each pair probability on W
	M = np.apply_along_axis(lambda ij_pair: W.loc[ij_pair[0],ij_pair[1]], 2, M)

	# now we actually have "M"
	M = pd.DataFrame(M)

	return M


def encode_features(ticker, itr_date, n_states, timespan, modelname):
	_result = []

	features_df = load_scenarioc_features(ticker, parseDate=False)
	features_df.set_index("Date", inplace=True)

	_idx = features_df.index.get_loc(itr_date)

	if _idx >= timespan:
		for itr_serie in features_df.columns.tolist():
			mtf = markov_transition_field(features_df.iloc[_idx-timespan:_idx][itr_serie], n_states, features_df[itr_serie].min(), features_df[itr_serie].max())
			_result.append(mtf.values)

		_result = np.array(_result)
		_result = _result.reshape(_result.shape[1],_result.shape[2],_result.shape[0])
	else:
		_result = None


	return _result


def calc_features(raw_df, verbose=True):
	result_df = pd.DataFrame()

	result_df["Close"] = raw_df["Close"]
	result_df["RSI_60"] = vectorized_funs.rsiFunc(raw_df["Close"], 60)
	cmf, dmf = vectorized_funs.calc_chaikin_money_flow(raw_df, window=21)
	result_df["CHAIKIN_MFLOW_21"] = cmf

	## CHANGE THIS TO MEAN?
	result_df.where(~np.isnan(result_df), TINY_FLOAT, inplace=True)
	result_df.where(-np.isinf(result_df), TINY_FLOAT, inplace=True)

	return result_df



def calc_labels(raw_df, verbose=True):
	result_df = pd.DataFrame()

	result_df["RETURN_1"] = ((raw_df["Close"] / raw_df["Close"].shift(1)) - 1)
	result_df["RETURN_30"] = ((raw_df["Close"] / raw_df["Close"].shift(30)) - 1)
	result_df["RETURN_60"] = ((raw_df["Close"] / raw_df["Close"].shift(60)) - 1)

	result_df["RETURN_1"] = result_df["RETURN_1"].shift(-1)
	result_df["RETURN_30"] = result_df["RETURN_30"].shift(-30)
	result_df["RETURN_60"] = result_df["RETURN_60"].shift(-60)

	result_df = np.around(result_df, 4)

	result_df.where(~np.isnan(result_df), TINY_FLOAT, inplace=True)
	result_df.where(-np.isinf(result_df), TINY_FLOAT, inplace=True)

	return result_df


def prepare_problemspace(ticker_list, timespan, bins):
	## Load the labels for every ticker
	_labels = dict()

	for itr_ticker in ticker_list:
		_labels[itr_ticker] = load_scenarioc_labels(itr_ticker, True)
		_labels[itr_ticker].set_index("Date", inplace=True)

	## Look into disk and list all TICKERS and DATES encodings
	listing = glob.glob('./data/D_TRIALA/*_{}_{}.npy'.format(timespan, bins))
	rx = "MTFIELD_(.*)_(\d{4}-\d{2}-\d{2})_" + str(timespan) + "_" + str(bins) + ".npy".format(timespan, bins)
	_tickers = np.array([ re.search(rx, x).group(1) for x in listing ])
	_dates = np.array([ pd.to_datetime(re.search(rx, x).group(2)) for x in listing ])

	return _tickers, _dates, _labels


def create_model(input_shape=(224,224,3), filter_shape=(3, 3), output_size=3, FC_layers=6):
	model = Sequential()

	# Block 1
	model.add(Conv2D(64, filter_shape, activation='relu', padding='same', name='block1_conv1', input_shape=input_shape))
	model.add(Conv2D(64, filter_shape, activation='relu', padding='same', name='block1_conv2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

	# Block 2
	model.add(Conv2D(128, filter_shape, activation='relu', padding='same', name='block2_conv1'))
	model.add(Conv2D(128, filter_shape, activation='relu', padding='same', name='block2_conv2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

	# Block 3
	model.add(Conv2D(256, filter_shape, activation='relu', padding='same', name='block3_conv1'))
	model.add(Conv2D(256, filter_shape, activation='relu', padding='same', name='block3_conv2'))
	model.add(Conv2D(256, filter_shape, activation='relu', padding='same', name='block3_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

	# Block 4
	model.add(Conv2D(512, filter_shape, activation='relu', padding='same', name='block4_conv1'))
	model.add(Conv2D(512, filter_shape, activation='relu', padding='same', name='block4_conv2'))
	model.add(Conv2D(512, filter_shape, activation='relu', padding='same', name='block4_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

	# Block 5
	model.add(Conv2D(512, filter_shape, activation='relu', padding='same', name='block5_conv1'))
	model.add(Conv2D(512, filter_shape, activation='relu', padding='same', name='block5_conv2'))
	model.add(Conv2D(512, filter_shape, activation='relu', padding='same', name='block5_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))


	# FC
	model.add(Flatten())

	for i in range(FC_layers):
		model.add(Dense(4096, activation='relu', kernel_initializer="uniform"))

	#model.add(Dense(output_size, kernel_initializer='normal'))
	model.add(Dense(output_size, kernel_initializer='normal'))

	model.compile(loss='mean_squared_error', optimizer='adam')

	return model


def seq_data(dates, tickers, labels, timespan, bins):
	X = None
	y = None
	
	for i in range(dates.shape[0]):
		_iter = (dates[i], tickers[i])

		X = load_scenarioc_encodings(_iter[1], datetime.date.strftime(_iter[0], "%Y-%m-%d"), timespan, bins)
		y = labels[_iter[1]].loc[_iter[0]].values
		
		## when X.shape is an empty tuple, we don't yield and go to the next iteration
		try:
			if X.shape:
				yield X, y
		except AttributeError:
			print("scenarioc.seq_data - X came NoneType", _iter, "\n")


def seq_batch(dates, tickers, labels, timespan, bins, batch_size=32):
	seq = seq_data(dates, tickers, labels, timespan, bins)
	X = []
	y = []
	while True:
		X = []
		y = []
		for i in range(batch_size):
			try:
				_r, _rr = next(seq)
				X.append(_r)
				y.append(_rr)
			except StopIteration:
				yield X, y
				raise StopIteration
		
		yield X, y

def features_stats(dates, tickers, labels, timespan, bins):
	_r = []
	data_iterator = seq_batch(dates, tickers, labels, timespan, bins)

	try:
		while True:
			X_batch, y_batch = next(data_iterator)
			X_batch = np.array(X_batch)
			_r.append([X_batch.mean(), X_batch.var()])

	except StopIteration:
		_r = np.array(_r)

	return _r[:,0].mean(), np.sqrt(np.mean(_r[:,1]))


def train(model, dates, tickers, labels, timespan, bins, features_mean, features_std):
	data_iterator = seq_batch(dates, tickers, labels, timespan, bins)
	try:
		while True:
			X_batch, y_batch = next(data_iterator)
			X_batch = np.array(X_batch)

			y_batch = np.where(~np.isnan(y_batch), y_batch, 0.0)
			y_batch = np.where(~np.isinf(y_batch), y_batch, 0.0)

			y_batch = y_batch[ : , :3]

			## Normalize Features
			X_batch = (X_batch - features_mean) / features_std

			model.fit(X_batch, y_batch[:,1], epochs=1, batch_size=32, verbose=0)

	except StopIteration:
		v = None
	
	return model

def evaluate(model, dates, tickers, labels, timespan, bins, features_mean, features_std):
	_r = dict()
	y_true = []
	y_pred = []

	data_iterator = seq_batch(dates, tickers, labels, timespan, bins)

	try:
		while True:
			X_batch, y_batch = next(data_iterator)
			X_batch = np.array(X_batch)

			y_batch = np.where(~np.isnan(y_batch), y_batch, 0.0)
			y_batch = np.where(~np.isinf(y_batch), y_batch, 0.0)

			y_batch = y_batch[ : , :3]

			## Normalize Features
			X_batch = (X_batch - features_mean) / features_std

			y_pred.append(model.predict(X_batch, verbose=0))
			y_true.append(y_batch[:,1])

	except StopIteration:
		v = None

	y_true = np.concatenate(y_true)
	y_pred = np.concatenate(y_pred)

	gain_true = (y_true > 0.0) * 1.0
	gain_pred = (y_pred > 0.0) * 1.0

	_r["mse"] = mean_squared_error(y_true, y_pred)
	_r["r_squared"] = r2_score(y_true, y_pred, multioutput = "uniform_average")
	_r["accuracy"] = accuracy_score(gain_true, gain_pred)

	return _r


def store_scenarioc_features(features_df, ticker):
	features_df.to_csv("{}/{}_scenarioc_X.csv".format(paths.TRIALA_DATA_PATH, ticker))
	return True

def load_scenarioc_features(ticker, parseDate=True):
	_r = None

	try:
		_r = pd.read_csv("{}/{}_scenarioc_X.csv".format(paths.TRIALA_DATA_PATH, ticker))

		if parseDate:
			_r["Date"] = pd.to_datetime(_r["Date"], infer_datetime_format=True)

	except:
		_r = None

	return _r

def check_encoding_exists(ticker, date, timespan, bins):
	return os.path.isfile("{}/MTFIELD_{}_{}_{}_{}.npy".format(paths.TRIALA_DATA_PATH, ticker, date, timespan, bins))

def store_scenarioc_encodings(feature_data, ticker, date, timespan, bins):
	np.save("{}/MTFIELD_{}_{}_{}_{}.npy".format(paths.TRIALA_DATA_PATH, ticker, date, timespan, bins), feature_data)
	return True

def load_scenarioc_encodings(ticker, date, timespan, bins):
	_r = None

	try:
		_r = np.load("{}/MTFIELD_{}_{}_{}_{}.npy".format(paths.TRIALA_DATA_PATH, ticker, date, timespan, bins))
	except:
		_r = None

	return _r

def store_scenarioc_labels(features_df, ticker):
	features_df.to_csv("{}/{}_scenarioc_Y.csv".format(paths.TRIALA_DATA_PATH, ticker))

	return True

def load_scenarioc_labels(ticker, parseDate=True):
	_r = None

	try:
		_r = pd.read_csv("{}/{}_scenarioc_Y.csv".format(paths.TRIALA_DATA_PATH, ticker))

		if parseDate:
			_r["Date"] = pd.to_datetime(_r["Date"], infer_datetime_format=True)

	except:
		_r = None

	return _r

