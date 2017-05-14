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

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

import itertools


import math

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
#

def gramian_anguler_field(serie, min_val, max_val):

	n = serie.shape[0]
	min_val = serie.min() if min_val is None else min_val
	max_val = serie.max() if max_val is None else max_val

	serie_normalized = ((serie - max_val) + (serie - min_val())) / (max_val - min_val())

	polar_teta = np.arccos(serie_normalized)
	polar_r = np.arange(serie_normalized.shape[0]) / n

	G = np.cos(np.add(np.tile((serie).values.reshape(n,1), n), (serie).values.reshape(n, 1)))

	return G


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


def transform_features(ticker, itr_date, n_states, timespan, modelname):
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

	daily_return = ((raw_df["Close"] / raw_df["Close"].shift(1)) - 1)

	result_df["OBV"] = vectorized_funs.onbalancevolumeFunc(daily_return, raw_df["Volume"])

	return result_df



def calc_labels(raw_df, verbose=True):
	return scenarioa.calc_labels(raw_df, verbose)



def prepare_problemspace(ticker_list, date_from, date_until, model_name, normalize=True, return_type="pandas"):
	X = []
	y = []

	labels_df = None
	labels_ticker = None
	date_idx = -1
	current_encoding = None
	ticker_hasdate = None
	encoding_exists = None
	enconding_ctx = []

	tickers = ticker_list

	dates = pd.date_range(date_from, date_until)

	_todo = [tickers, dates]
	#_todo = [x for x in itertools.product(*_todo)]

	for _it in itertools.product(*_todo):
		itr = dict(zip(["ticker", "date"], _it))

		if labels_ticker != itr["ticker"]:
			labels_ticker = itr["ticker"]
			labels_df = load_scenarioc_labels(itr["ticker"], parseDate=True)
			labels_df.set_index("Date", inplace=True)

		try:
			date_idx = labels_df.index.get_loc(itr["date"])
			ticker_hasdate = True
		except KeyError:
			ticker_hasdate = False

		encoding_exists = False

		if ticker_hasdate:
			current_encoding = load_scenarioc_encodings(itr["ticker"], model_name, datetime.datetime.strftime(itr["date"], "%Y-%m-%d"))
			encoding_exists = False if (len(current_encoding.shape) == 0) else True

			if encoding_exists:
				enconding_ctx.append(_it)
				X.append(current_encoding)
				y.append(labels_df.iloc[date_idx])

	X = np.array(X)
	y = np.array(y)

	y = np.where(~np.isnan(y),y, 0.0)
	y = np.where(~np.isinf(y),y, 0.0)


	return X, y, enconding_ctx


def create_model(side, channels):
	model = Sequential()

	model.add(Conv2D(64, (3, 3), input_shape=(side, side, channels), activation="relu"))
	kutils.ConvBlock(1, 64, model, add_maxpooling=True)
	kutils.ConvBlock(2, 128, model, add_maxpooling=True)
	kutils.ConvBlock(3, 256, model, add_maxpooling=False)

	kutils.ConvBlock(3, 512, model, add_maxpooling=False)
	kutils.ConvBlock(3, 512, model, add_maxpooling=True)

	model.add(Flatten())

	kutils.FCBlock(model, add_batchnorm=True, add_dropout=True)
	kutils.FCBlock(model, add_batchnorm=True, add_dropout=True)

	model.add(Dense(4, kernel_initializer='normal'))

	# Compile model
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	return model




def fit(model, X_reduced, y_train, nb_epoch=1):


		## Create a new dimension for the "channel"
		X = X_reduced

		## Fit
		model.fit(X, y_train, epochs=nb_epoch, batch_size=128, verbose=1)

		return model


def evaluate(model, X_test, y_test, return_type="dict"):
	_r = dict()

	X = X_test

	y_pred = model.predict(X, verbose=0)
	gain_test = (y_test > 0.0) * 1.0
	gain_pred = (y_pred > 0.0) * 1.0

	_r["r_squared"] = r2_score(y_test, y_pred, multioutput = "uniform_average")
	_r["accuracy"] = accuracy_score(gain_test, gain_pred)

	if return_type == "pandas":
		_r["r_squared"] = [_r["r_squared"]]
		_r["accuracy"] = [_r["accuracy"]]
		_r = pd.DataFrame.from_dict(_r)

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

def store_scenarioc_encodings(feature_data, ticker, modelname, date):
	np.save("{}/MTFIELD_{}_{}_{}.npy".format(paths.TRIALA_DATA_PATH, ticker, modelname, date), feature_data)
	return True

def load_scenarioc_encodings(ticker, modelname, date):
	_r = None

	try:
		_r = np.load("{}/MTFIELD_{}_{}_{}.npy".format(paths.TRIALA_DATA_PATH, ticker, modelname, date))
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

