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
from sklearn.metrics import accuracy_score

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


def calc_features(raw_df, verbose=True, normalize=False):
	return scenarioa.calc_features(raw_df, verbose, normalize)



def calc_labels(raw_df, verbose=True):
	return scenarioa.calc_labels(raw_df, verbose)



def prepare_problemspace(target_ticker, ticker_list, train_from, train_until, test_from, normalize=True, return_type="pandas"):
	X_train_dict = dict()
	y_train_pnl = dict()
	y_train_dict = dict()
	X_test_dict = dict()
	y_test_pnl = dict()
	y_test_dict = dict()

	itr_df = None
	X_train_pnl = None
	y_train_pnl = None
	X_test_pnl = None
	y_test_pnl = None

	for itr_ticker in ticker_list:
		## Load Features
		itr_df = load_scenariob_features(itr_ticker, parseDate=True)
		itr_df.set_index("Date", inplace=True)

		# Split Features into Train and Test
		X_train_dict[itr_ticker] = itr_df.loc[train_from:train_until, :]
		X_test_dict[itr_ticker] = itr_df.loc[test_from:, :]

		## Load Labels
		itr_df = load_scenariob_labels(itr_ticker, parseDate=True)
		itr_df.set_index("Date", inplace=True)

		# Split Features into Train and Test
		y_train_dict[itr_ticker] = itr_df.loc[train_from:train_until, :]
		y_test_dict[itr_ticker] = itr_df.loc[test_from:, :]

	# Create Panel for Train Features
	X_train_pnl = pd.Panel(X_train_dict)
	X_train_pnl = X_train_pnl.swapaxes(0,1).swapaxes(2,1)

	# Create Panel for Test Features
	X_test_pnl = pd.Panel(X_test_dict)
	X_test_pnl = X_test_pnl.swapaxes(0,1).swapaxes(2,1)


	# Create Panel for Train Labels
	y_train_pnl = pd.Panel(y_train_dict)
	y_train_pnl = y_train_pnl.swapaxes(0,1)
	y_train_pnl = y_train_pnl.to_frame(filter_observations=False).T

	# Create Panel for Test Labels
	y_test_pnl = pd.Panel(y_test_dict)
	y_test_pnl = y_test_pnl.swapaxes(0,1)
	y_test_pnl = y_test_pnl.to_frame(filter_observations=False).T


	## Normalize Close
	if normalize:
		X_train_pnl.loc[:,"Close",:] = X_train_pnl.loc[:,"Close",:] / X_train_pnl.loc[:,"Close",:].max().max()
		X_test_pnl.loc[:,"Close",:] = X_test_pnl.loc[:,"Close",:] / X_train_pnl.loc[:,"Close",:].max().max()


	# Prepare output when necessary
	if return_type == "numpy":
		X_train_pnl = X_train_pnl.values
		X_test_pnl = X_test_pnl.values
		y_train_pnl = y_train_pnl.values
		y_test_pnl = y_test_pnl.values

		X_train_pnl = np.where(~np.isnan(X_train_pnl),X_train_pnl, 0.0)
		X_train_pnl = np.where(~np.isinf(X_train_pnl),X_train_pnl, 0.0)

		X_test_pnl = np.where(~np.isnan(X_test_pnl),X_test_pnl, 0.0)
		X_test_pnl = np.where(~np.isinf(X_test_pnl),X_test_pnl, 0.0)

		y_train_pnl = np.where(~np.isnan(y_train_pnl),y_train_pnl, 0.0)
		y_train_pnl = np.where(~np.isinf(y_train_pnl),y_train_pnl, 0.0)

		y_test_pnl = np.where(~np.isnan(y_test_pnl),y_test_pnl, 0.0)
		y_test_pnl = np.where(~np.isinf(y_test_pnl),y_test_pnl, 0.0)


	return X_train_pnl, y_train_pnl, X_test_pnl, y_test_pnl


def create_model(n_tickers):
	model = Sequential()

	model.add(Conv2D(64, (3, 3), input_shape=(29, n_tickers, 1), activation="relu"))
	kutils.ConvBlock(1, 64, model, add_maxpooling=False)
	kutils.ConvBlock(2, 128, model, add_maxpooling=False)
	kutils.ConvBlock(3, 256, model, add_maxpooling=False)

	kutils.ConvBlock(3, 512, model, add_maxpooling=True)
	kutils.ConvBlock(3, 512, model, add_maxpooling=True)

	model.add(Flatten())

	kutils.FCBlock(model, add_batchnorm=True)
	kutils.FCBlock(model, add_batchnorm=True)
	kutils.FCBlock(model, add_batchnorm=True)

	model.add(Dense(4 * n_tickers, kernel_initializer='normal'))

	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model



def fit(model, X_train, y_train, nb_epoch=1):


		## Create a new dimension for the "channel"
		X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

		## Fit
		model.fit(X_train, y_train, epochs=nb_epoch, batch_size=32, verbose=1)

		return model


def evaluate(model, X_test, y_test, return_type="dict"):
	_r = dict()

	y_pred = model.predict(X_test, verbose=0)
	gain_test = (y_test > 0.0) * 1.0
	gain_pred = (y_pred > 0.0) * 1.0

	_r["r_squared"] = r2_score(y_test, y_pred, multioutput = "uniform_average")
	_r["accuracy"] = accuracy_score(gain_test, gain_pred)

	if return_type == "pandas":
		_r["r_squared"] = [_r["r_squared"]]
		_r["accuracy"] = [_r["accuracy"]]
		_r = pd.DataFrame.from_dict(_r)

	return _r


def store_scenariob_features(features_df, ticker):
	features_df.to_csv("{}/{}_scenariob_X.csv".format(paths.TRIALA_DATA_PATH, ticker))
	return True

def load_scenariob_features(ticker, parseDate=True):
	_r = None

	try:
		_r = pd.read_csv("{}/{}_scenariob_X.csv".format(paths.TRIALA_DATA_PATH, ticker))

		if parseDate:
			_r["Date"] = pd.to_datetime(_r["Date"], infer_datetime_format=True)

	except:
		_r = None

	return _r

def store_scenariob_labels(features_df, ticker):
	features_df.to_csv("{}/{}_scenariob_Y.csv".format(paths.TRIALA_DATA_PATH, ticker))

	return True

def load_scenariob_labels(ticker, parseDate=True):
	_r = None

	try:
		_r = pd.read_csv("{}/{}_scenariob_Y.csv".format(paths.TRIALA_DATA_PATH, ticker))

		if parseDate:
			_r["Date"] = pd.to_datetime(_r["Date"], infer_datetime_format=True)

	except:
		_r = None

	return _r

