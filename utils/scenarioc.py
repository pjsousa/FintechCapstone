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

_dd = None

def accuracy_gainloss(y_true, y_pred):
    _dd = [K.eval(y_true), K.eval(y_pred)]
    gain_test = K.cast(K.greater(y_true, K.constant(0.5)), K.floatx())
    gain_pred = K.cast(K.greater(y_pred, K.constant(0.5)), K.floatx())

    return binary_accuracy(gain_test, gain_pred)


def r2_regression(y_true, y_pred):
	numerator = K.square(y_true - y_pred)
	denominator = K.sum(K.square(y_true - K.mean(y_true)))

	return K.constant(1.0) - (numerator / denominator)



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
	_full = scenarioa.calc_features(raw_df, verbose)

	result_df["Close"] = _full["Close"]

	# result_df["BOLL_60_UP"] = _full["BOLL_60_UP"]
	# result_df["BOLL_60_DOWN"] = _full["BOLL_60_DOWN"]

	# result_df["MACD"] = _full["MACD"]
	# result_df["MACD_EMASLOW"] = _full["MACD_EMASLOW"]
	# result_df["MACD_EMAFAST"] = _full["MACD_EMAFAST"]

	result_df["RSI_60"] = _full["RSI_60"]

	# result_df["ADX"] = _full["ADX"]
	# result_df["ADX_PDI"] = _full["ADX_PDI"]
	# result_df["ADX_NDI"] = _full["ADX_NDI"]

	# result_df["AROONUP_20"] = _full["AROONUP_20"]
	# result_df["AROONDOWN_20"] = _full["AROONDOWN_20"]

	result_df["CHAIKIN_MFLOW_21"] = _full["CHAIKIN_MFLOW_21"]
	# result_df["DAILY_MFLOW_21"] = _full["DAILY_MFLOW_21"]

	#result_df["OBV"] = _full["OBV"]


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
			try:
				current_encoding = load_scenarioc_encodings(itr["ticker"], model_name, datetime.datetime.strftime(itr["date"], "%Y-%m-%d"))
				encoding_exists = False if (current_encoding is None) or (len(current_encoding.shape) == 0) else True
			except:
				print("Error", _it)

			if encoding_exists:
				enconding_ctx.append(_it)
				X.append(current_encoding[:,:,:3])
				y.append(labels_df.iloc[date_idx])

	X = np.array(X)
	y = np.array(y)

	y = np.where(~np.isnan(y),y, 0.0)
	y = np.where(~np.isinf(y),y, 0.0)


	return X, y, enconding_ctx


def create_model(side, channels, output_shape=4):
	model = Sequential()

	model.add(Conv2D(64, (3, 3), input_shape=(side, side, channels), activation="relu"))
	kutils.ConvBlock(1, 64, model, add_maxpooling=True)
	kutils.ConvBlock(2, 128, model, add_maxpooling=True)
	kutils.ConvBlock(3, 256, model, add_maxpooling=False)

	kutils.ConvBlock(3, 512, model, add_maxpooling=False)
	kutils.ConvBlock(3, 512, model, add_maxpooling=True)

	model.add(Flatten())

	kutils.FCBlock(model, add_batchnorm=True, add_dropout=False)
	kutils.FCBlock(model, add_batchnorm=True, add_dropout=False)

	model.add(Dense(output_shape, kernel_initializer='normal'))

	# Compile model
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	return model

def create_model():
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(224,224,3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
    
    
    # FC
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer="uniform"))
    model.add(Dense(4096, activation='relu', kernel_initializer="uniform"))
    model.add(Dense(4, kernel_initializer='normal'))


    model.compile(loss='mean_squared_error', optimizer='adam')

    return model




def fit(model, X_train, y_train, X_test, y_test, nb_epoch=1):
	filepath=paths.TEMP_PATH + "/" + "weights-improvement-{epoch:04d}-{val_loss:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor="val_loss" , verbose=1, save_best_only=True,mode="auto" )
	early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

	valdata = (X_test, y_test)

	model.fit(X_train, y_train, epochs=nb_epoch, callbacks=[early_stop, checkpoint], validation_data=valdata, batch_size=32)


	return model


def evaluate(model, X_test, y_test, return_type="dict"):
	_r = dict()

	X = X_test

	y_pred = model.predict(X, verbose=1)
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

