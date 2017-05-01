import numpy as np
import pandas as pd
import datetime
from functools import partial
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from utils import paths_helper as paths
from utils import vectorized_funs
from utils import kerasutil as kutils

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
	result_df = pd.DataFrame()

	# if verbose:
	# 	self.print_verbose_start()
	result_df["Close"] = raw_df["Close"]

	result_df["SMA_5"] = raw_df["Close"].rolling(window=5).mean()
	result_df["SMA_30"] = raw_df["Close"].rolling(window=30).mean()
	result_df["SMA_60"] = raw_df["Close"].rolling(window=60).mean()
	result_df["SMA_200"] = raw_df["Close"].rolling(window=200).mean()

	# if verbose:
	# 	self.print_verbose("SMAs")

	result_df["BOLL_5_UP"] = result_df["SMA_5"] + (2 * raw_df["Close"].rolling(window=5).std())
	result_df["BOLL_5_DOWN"] = result_df["SMA_5"] - (2 * raw_df["Close"].rolling(window=5).std())
	result_df["BOLL_30_UP"] = result_df["SMA_30"] + (2 * raw_df["Close"].rolling(window=30).std())
	result_df["BOLL_30_DOWN"] = result_df["SMA_30"] - (2 * raw_df["Close"].rolling(window=30).std())
	result_df["BOLL_60_UP"] = result_df["SMA_60"] + (2 * raw_df["Close"].rolling(window=60).std())
	result_df["BOLL_60_DOWN"] = result_df["SMA_60"] - (2 * raw_df["Close"].rolling(window=60).std())
	result_df["BOLL_200_UP"] = result_df["SMA_200"] + (2 * raw_df["Close"].rolling(window=200).std())
	result_df["BOLL_200_DOWN"] = result_df["SMA_200"] - (2 * raw_df["Close"].rolling(window=200).std())

	# if verbose:
	# 	self.print_verbose("BOLLINGER")

	emaslow, emafast, macd = vectorized_funs.calc_macd(raw_df["Close"], 26, 12)
	result_df["MACD"] = macd
	result_df["MACD_EMASLOW"] = emaslow
	result_df["MACD_EMAFAST"] = emafast

	# if verbose:
	# 	self.print_verbose("MACD")
	
	result_df["RSI_14"] = vectorized_funs.rsiFunc(raw_df["Close"], 14)
	result_df["RSI_21"] = vectorized_funs.rsiFunc(raw_df["Close"], 21)
	result_df["RSI_60"] = vectorized_funs.rsiFunc(raw_df["Close"], 60)

	# if verbose:
	# 	self.print_verbose("RSI")

	result_df["STOCOSCILATOR_14"] = vectorized_funs.calc_stochasticoscilator(raw_df, 14)
	result_df["STOCOSCILATOR_14_SMA"] = result_df["STOCOSCILATOR_14"].rolling(window=3).mean()

	# if verbose:
	# 	self.print_verbose("OSCILATOR")
	
	adx, pdi, ndi = vectorized_funs.calc_adx(raw_df)
	result_df["ADX"] = adx
	result_df["ADX_PDI"] = pdi
	result_df["ADX_NDI"] = ndi

	# if verbose:
	# 	self.print_verbose("ADX")

	aroon_up, aroon_down = vectorized_funs.calc_aroon(raw_df, 20)
	result_df["AROONUP_20"] = aroon_up.values
	result_df["AROONDOWN_20"] = aroon_down.values

	# if verbose:
	# 	self.print_verbose("AROON")

	cmf, dmf = vectorized_funs.calc_chaikin_money_flow(raw_df, window=21)
	result_df["CHAIKIN_MFLOW_21"] = cmf
	result_df["DAILY_MFLOW_21"] = dmf

	# if verbose:
	# 	self.print_verbose("CMFLOW")


	daily_return = ((raw_df["Close"] / raw_df["Close"].shift(1)) - 1)

	result_df["OBV"] = vectorized_funs.onbalancevolumeFunc(daily_return, raw_df["Volume"])

	if normalize:
		result_df.where(~np.isnan(result_df), 0.0, inplace=True)
		result_df.where(-np.isinf(result_df), 0.0, inplace=True)
		result_df = normalize_features(result_df)

	# if verbose:
	# 	self.print_verbose("OBV")

	return result_df



def calc_labels(raw_df, verbose=True):
	trial_a = pd.DataFrame()

	trial_a["RETURN_1"] = ((raw_df["Close"] / raw_df["Close"].shift(1)) - 1)
	trial_a["RETURN_30"] = ((raw_df["Close"] / raw_df["Close"].shift(30)) - 1)
	trial_a["RETURN_60"] = ((raw_df["Close"] / raw_df["Close"].shift(60)) - 1)
	trial_a["RETURN_200"] = ((raw_df["Close"] / raw_df["Close"].shift(200)) - 1)

	trial_a["RETURN_1"] = trial_a["RETURN_1"].shift(-1)
	trial_a["RETURN_30"] = trial_a["RETURN_30"].shift(-30)
	trial_a["RETURN_60"] = trial_a["RETURN_60"].shift(-60)
	trial_a["RETURN_200"] = trial_a["RETURN_200"].shift(-200)

	# if verbose:
	# 	self.print_verbose("RETURNS")
	# 	self.print_verbose_end()

	return trial_a



def prepare_problemspace(target_ticker, ticker_list, train_from, train_until, test_from, normalize=True, return_type="pandas"):
	X_train_dict = dict()
	y_train_df = dict()
	X_test_dict = dict()
	y_test_df = dict()
	itr_df = None
	X_train_pnl = None
	y_train_pnl = None
	X_test_pnl = None
	y_test_pnl = None

	for itr_ticker in ticker_list:
		## Load Features
		itr_df = load_scenarioa_features(itr_ticker, parseDate=True)
		itr_df.set_index("Date", inplace=True)

		# Split Features into Train and Test
		X_train_dict[itr_ticker] = itr_df.loc[train_from:train_until, :]
		X_test_dict[itr_ticker] = itr_df.loc[test_from:, :]

	# Create Panel for Train Features
	X_train_pnl = pd.Panel(X_train_dict)
	X_train_pnl = X_train_pnl.swapaxes(0,1).swapaxes(2,1)


	# Create Panel for Test Features
	X_test_pnl = pd.Panel(X_test_dict)
	X_test_pnl = X_test_pnl.swapaxes(0,1).swapaxes(2,1)


	## LABELS
	itr_df = load_scenarioa_labels(target_ticker, parseDate=True)
	itr_df.set_index("Date", inplace=True)

	# Split Labels into Train and Test
	y_train_df = itr_df.loc[train_from:train_until, :]
	y_test_df = itr_df.loc[test_from:, :]

	X_train_pnl = X_train_pnl.loc[y_train_df.index.tolist(),::]
	X_test_pnl = X_test_pnl.loc[y_test_df.index.tolist(),::]

	## Normalize Close
	if normalize:
		X_train_pnl.loc[:,"Close",:] = X_train_pnl.loc[:,"Close",:] / X_train_pnl.loc[:,"Close",:].max().max()
		X_test_pnl.loc[:,"Close",:] = X_test_pnl.loc[:,"Close",:] / X_train_pnl.loc[:,"Close",:].max().max()

	# Prepare output when necessary
	if return_type == "numpy":
		X_train_pnl = X_train_pnl.values
		X_test_pnl = X_test_pnl.values
		y_train_df = y_train_df.values
		y_test_df = y_test_df.values

		X_train_pnl = np.where(~np.isnan(X_train_pnl),X_train_pnl, 0.0)
		X_train_pnl = np.where(~np.isinf(X_train_pnl),X_train_pnl, 0.0)

		X_test_pnl = np.where(~np.isnan(X_test_pnl),X_test_pnl, 0.0)
		X_test_pnl = np.where(~np.isinf(X_test_pnl),X_test_pnl, 0.0)

		y_train_df = np.where(~np.isnan(y_train_df),y_train_df, 0.0)
		y_train_df = np.where(~np.isinf(y_train_df),y_train_df, 0.0)

		y_test_df = np.where(~np.isnan(y_test_df),y_test_df, 0.0)
		y_test_df = np.where(~np.isinf(y_test_df),y_test_df, 0.0)

	return X_train_pnl, y_train_df, X_test_pnl, y_test_df



def normalize_features(features_df):
	result_df = pd.DataFrame()
	result_df["Close"] = features_df["Close"]

	result_df["SMA_5"] = features_df["SMA_5"] / features_df["Close"]
	result_df["SMA_30"] = features_df["SMA_30"] / features_df["Close"]
	result_df["SMA_60"] = features_df["SMA_60"] / features_df["Close"]
	result_df["SMA_200"] = features_df["SMA_200"] / features_df["Close"]

	result_df["BOLL_5_UP"] = features_df["BOLL_5_UP"] / features_df["Close"]
	result_df["BOLL_5_DOWN"] = features_df["BOLL_5_DOWN"] / features_df["Close"]
	result_df["BOLL_30_UP"] = features_df["BOLL_30_UP"] / features_df["Close"]
	result_df["BOLL_30_DOWN"] = features_df["BOLL_30_DOWN"] / features_df["Close"]
	result_df["BOLL_60_UP"] = features_df["BOLL_60_UP"] / features_df["Close"]
	result_df["BOLL_60_DOWN"] = features_df["BOLL_60_DOWN"] / features_df["Close"]
	result_df["BOLL_200_UP"] = features_df["BOLL_200_UP"] / features_df["Close"]
	result_df["BOLL_200_DOWN"] = features_df["BOLL_200_DOWN"] / features_df["Close"]

	macd_scaler = MinMaxScaler().fit_transform(features_df.loc[:, ["MACD", "MACD_EMASLOW", "MACD_EMAFAST"]])
	result_df["MACD"] = pd.Series(macd_scaler.T[0], index=result_df.index)
	result_df["MACD_EMASLOW"] = pd.Series(macd_scaler.T[1], index=result_df.index)
	result_df["MACD_EMAFAST"] = pd.Series(macd_scaler.T[2], index=result_df.index)

	result_df["RSI_14"] = features_df["RSI_14"] / 100.0
	result_df["RSI_21"] = features_df["RSI_21"] / 100.0
	result_df["RSI_60"] = features_df["RSI_60"] / 100.0

	result_df["STOCOSCILATOR_14"] = features_df["STOCOSCILATOR_14"] / 100.0
	result_df["STOCOSCILATOR_14_SMA"] = features_df["STOCOSCILATOR_14_SMA"] / features_df["STOCOSCILATOR_14"]
	
	adx_scaler = MinMaxScaler().fit_transform(features_df.loc[:, ["ADX", "ADX_PDI", "ADX_NDI"]])
	result_df["ADX"] = pd.Series(adx_scaler.T[0], index=result_df.index)
	result_df["ADX_PDI"] = pd.Series(adx_scaler.T[1], index=result_df.index)
	result_df["ADX_NDI"] = pd.Series(adx_scaler.T[2], index=result_df.index)

	result_df["AROONUP_20"] = features_df["AROONUP_20"] / 100.0
	result_df["AROONDOWN_20"] = features_df["AROONDOWN_20"] / 100.0

	cmf_scaler = MinMaxScaler().fit_transform(features_df.loc[:, ["CHAIKIN_MFLOW_21", "DAILY_MFLOW_21"]])
	result_df["CHAIKIN_MFLOW_21"] = pd.Series(cmf_scaler.T[0], index=result_df.index)
	result_df["DAILY_MFLOW_21"] = pd.Series(cmf_scaler.T[1], index=result_df.index)

	result_df["OBV"] = features_df["OBV"] / result_df["Close"]

	## Let's clean again the nan and inf 
	#  Because Some indicators might be 0.0 and produce Inf when they are on the denominator.
	result_df.where(~np.isnan(result_df), 0.0, inplace=True)
	result_df.where(-np.isinf(result_df), 0.0, inplace=True)

	assert np.sum(~features_df.columns.to_series().isin(result_df.columns.to_series())) == 0, \
		"result_df and features_df must have the same column set"

	return result_df



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

	model.add(Dense(4, kernel_initializer='normal'))

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

	X = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

	y_pred = model.predict(X, verbose=0)
	gain_test = (y_test > 0.0) * 1.0
	gain_pred = (y_pred > 0.0) * 1.0

	_r["r_squared"] = r2_score(y_test, y_pred, multioutput = "uniform_average")
	_r["accuracy"] = accuracy_score(gain_test, gain_pred)

	if return_type == "pandas":
		_r["r_squared"] = [_r["r_squared"]]
		_r["accuracy"] = [_r["accuracy"]]
		_r = pd.DataFrame.from_dict(_r)

	del X

	return _r




def store_scenarioa_features(features_df, ticker):
	features_df.to_csv("{}/{}_scenarioa_X.csv".format(paths.TRIALA_DATA_PATH, ticker))
	return True

def load_scenarioa_features(ticker, parseDate=True):
	_r = None

	try:
		_r = pd.read_csv("{}/{}_scenarioa_X.csv".format(paths.TRIALA_DATA_PATH, ticker))

		if parseDate:
			_r["Date"] = pd.to_datetime(_r["Date"], infer_datetime_format=True)

	except:
		_r = None

	return _r

def store_scenarioa_labels(features_df, ticker):
	features_df.to_csv("{}/{}_scenarioa_Y.csv".format(paths.TRIALA_DATA_PATH, ticker))

	return True

def load_scenarioa_labels(ticker, parseDate=True):
	_r = None

	try:
		_r = pd.read_csv("{}/{}_scenarioa_Y.csv".format(paths.TRIALA_DATA_PATH, ticker))

		if parseDate:
			_r["Date"] = pd.to_datetime(_r["Date"], infer_datetime_format=True)

	except:
		_r = None

	return _r

