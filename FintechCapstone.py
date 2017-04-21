from __future__ import division, print_function

import math
import uuid
import datetime
import pandas as pd
import numpy as np
from functools import partial
from dateutil import parser as dtparser

from utils import datafetch
from utils import datapipe
from utils import baseline_model
from utils import scenarioa
from utils import paths_helper as paths

import argparse
import os


TINY_FLOAT = 1e-128

class FinCapstone():

	def __init__(self,
				model_name="ExampleFintech",
				ticker_list_samplesize=100,
				path_dataload_result="RES_initial_dataload.csv",
				path_ticker_list=None,
				date_from='1900-01-01',
				date_to=str(datetime.date.today()),
				fill_value=1e-128,
				ticker_list=None,
				timespan=None,
				timespan_ab=None,
				train_from="2010-01-01",
				train_until="2015-12-31",
				test_from="2016-01-01",
				bins=None):

		self.ticker_list = None
		self.timespan = None
		self.timespan_ab = None
		self.bins = None
		self.ticker_list_samplesize = ticker_list_samplesize
		self.PATH_DATALOAD_RESULT = path_dataload_result
		self.date_from = date_from
		self.date_to = date_to
		self.fill_value = fill_value
		self.train_from = dtparser.parse(train_from)
		self.train_until = dtparser.parse(train_until)
		self.test_from = dtparser.parse(test_from)
		self.model_name = model_name


		self._start = None
		self._step_i = None
		self._step_f = None
		self._end = None

		if ticker_list is None:
			self.ticker_list = self.provision_fulltickerlist()
		else:
			self.ticker_list = ticker_list


		if timespan is None:
			self.timespan = {
				"short_term": [1,3,5]
				,"medium_term": [40, 60]
				,"long_term": [90, 150, 220]
			}
		else:
			self.timespan = timespan

		if timespan_ab is None:
			self.timespan_ab = {
				"short_term": [5]
				,"medium_term": [30]
				,"long_term": [20]
			}
		else:
			self.timespan_ab = timespan_ab

		if bins is None:
			self.bins = np.array([-100, -20, -10, -5, -1, 1, 5, 10, 20, 100]).astype("float32")
		else:
			self.bins = np.array(bins).astype("float32")


	def provision_fulltickerlist(self):
		_rr = None
		_rr = datafetch.load_exchangesinfos()
		#_rr = _rr.drop([2922])

		if type(self.ticker_list_samplesize) == int and self.ticker_list_samplesize > 0:
			self.ticker_list = _rr.iloc[0:self.ticker_list_samplesize]["Symbol"].values
		else:
			self.ticker_list = _rr["Symbol"].values

		return self.ticker_list

	def provision_validtickerlist(self):
		ticker_list = pd.read_csv(self.PATH_DATALOAD_RESULT)
		ticker_list = ticker_list[ticker_list["STATUS"] == "OK"]
		return ticker_list

	def run_initial_dataload(self):
		_ok = None
		_nok = None
		_r = None

		# Fetch data for all tickers
		_r = datafetch.initial_dataload(self.ticker_list, verbose=True, del_temp=True)

		# Store the results from the data fetching
		OKs = ["OK" for x in _r["OK"]]
		NOKs = ["NOK" for x in _r["NOK"]]

		_ok = pd.DataFrame([OKs, _r["OK"]]).T
		_nok = pd.DataFrame([NOKs, _r["NOK"]]).T

		_r = pd.concat([_ok, _nok], axis=0)
		_r.columns = ["STATUS", "Symbol"]
		_r.to_csv(self.PATH_DATALOAD_RESULT, index=False)

	def feature_engineering(self, feature_set="baseline"):
		## Only work with tickers that were successfull during datafetch
		ticker_list = self.provision_validtickerlist()
		work_df = None

		for ix, row in ticker_list.iterrows():
			itr_ticker = row["Symbol"]
			print("\n\n - {} - \n".format(itr_ticker))
			itr_df = datafetch.load_raw_frame(itr_ticker,parseDate=False, dropAdjClose=True)
			
			itr_df = itr_df[pd.to_datetime(itr_df["Date"]) >= dtparser.parse(self.date_from)]
			itr_df = itr_df[pd.to_datetime(itr_df["Date"]) < dtparser.parse(self.date_to)]


			if feature_set == "baseline":
				work_df = baseline_model.calc_features(itr_df,verbose=True)
				self.store_baseline_features(work_df, itr_ticker)

				work_df = baseline_model.calc_labels(itr_df, self.timespan, verbose=True)
				self.store_baseline_labels(work_df, itr_ticker)
			elif feature_set == "scenarioa":
				itr_df.set_index("Date", inplace=True)
				work_df = scenarioa.calc_features(itr_df, normalize=True, verbose=True)
				self.store_scenarioa_features(work_df, itr_ticker)

				work_df = scenarioa.calc_labels(itr_df, verbose=True)
				self.store_scenarioa_labels(work_df, itr_ticker)
			else:
				print("Invalid Feature Set.")

	def split_tickerlist(self, n_splits=4):
		ticker_list = pd.read_csv(self.PATH_DATALOAD_RESULT)
		_size = 1.0 / n_splits
		itr_ticker_list = None
		itr_path = None

		for idx in range(n_splits):
			_from = math.floor(idx * _size * ticker_list.shape[0])
			_to = math.floor((idx+1) * _size * ticker_list.shape[0])
			_to = min(_to, ticker_list.shape[0])

			itr_ticker_list = ticker_list.iloc[_from:_to]
			itr_path = self.PATH_DATALOAD_RESULT.replace(".csv", "_{}.csv".format(idx))

			itr_ticker_list.to_csv(itr_path, index=False)

	def valid_ticker_list(self):
		_r = self.provision_validtickerlist()["Symbol"].values
		return _r

	def train_baseline(self, nb_epoch=100):
		results = None
		X_train = None
		y_train = None
		X_test = None
		y_test = None

		results = np.zeros_like(self.valid_ticker_list())

		self.print_verbose_start()

		for idx_ticker, itr_ticker in enumerate(self.valid_ticker_list()):
			try:
				self.print_verbose("{}/{} - {}".format(idx_ticker, results.shape[0], itr_ticker))

				features_df = self.load_baseline_features(itr_ticker, parseDate=True)
				features_df.set_index("Date", inplace=True)

				labels_df = self.load_baseline_labels(itr_ticker, parseDate=True)
				labels_df.set_index("Date", inplace=True)
	
				model = baseline_model.create_model()
				X_train, y_train, X_test, y_test = baseline_model.prepare_problemspace(features_df, labels_df, self.train_from, self.train_until, self.test_from)

				baseline_model.fit(model, X_train, y_train, X_test, y_test, nb_epoch)

				results[idx_ticker] = baseline_model.evaluate(model, X_test, y_test)

				model.save_weights("{}/weights{}_{}_{}.h5".format(paths.TEMP_PATH, "baseline", self.model_name, itr_ticker))
			except:
				print("")


		self.print_verbose_end()

		return pd.DataFrame(data=results, index=self.valid_ticker_list())

	def predict_baseline(self, ticker):
		features_df = self.load_baseline_features(ticker, parseDate=True)
		features_df.set_index("Date", inplace=True)

		labels_df = self.load_baseline_labels(ticker, parseDate=True)
		labels_df.set_index("Date", inplace=True)

		model = baseline_model.create_model()
		X_train, y_train, X_test, y_test = baseline_model.prepare_problemspace(features_df, labels_df, self.train_from, self.train_until, self.test_from)
		model.load_weights("{}/weights{}_{}_{}.h5".format(TEMP_PATH, "baseline", self.model_name, ticker))

		y_pred = model.predict(X_test, verbose=0)

		return y_pred


	## Storage Helpers
	def store_baseline_features(self, features_df, ticker):
		features_df.to_csv("{}/{}_baseline_X.csv".format(paths.BASELINE_DATA_PATH, ticker), index=False)
		return True

	def load_baseline_features(self, ticker, parseDate=True):
		_r = None

		try:
			_r = pd.read_csv("{}/{}_baseline_X.csv".format(paths.BASELINE_DATA_PATH, ticker))

			if parseDate:
				_r["Date"] = pd.to_datetime(_r["Date"], infer_datetime_format=True)
		except:
			_r = None

		return _r

	def store_baseline_labels(self, features_df, ticker):
		features_df.to_csv("{}/{}_baseline_Y.csv".format(paths.BASELINE_DATA_PATH, ticker), index=False)

		return True

	def load_baseline_labels(self, ticker, parseDate=True):
		_r = None

		try:
			_r = pd.read_csv("{}/{}_baseline_Y.csv".format(paths.BASELINE_DATA_PATH, ticker))

			if parseDate:
				_r["Date"] = pd.to_datetime(_r["Date"], infer_datetime_format=True)
		except e:
			_r = None


		return _r

	def store_scenarioa_features(self, features_df, ticker):
		
		return scenarioa.store_scenarioa_features(features_df, ticker)

	def load_scenarioa_features(self, ticker, parseDate=True):
		
		return scenarioa.load_scenarioa_features(features_df, ticker)

	def store_scenarioa_labels(self, features_df, ticker):
		
		return scenarioa.store_scenarioa_labels(features_df, ticker)

	def load_scenarioa_labels(self, ticker, parseDate=True):
		
		return scenarioa.load_scenarioa_labels(features_df, ticker)

	## Verbose Helpers
	def reset_verboseclock(self):
		self._start = datetime.datetime.now()
		self._step_i = self._start

	def print_verbose_start(self):
		self.reset_verboseclock()
		print("| | START     - {}".format(str(self._start)))

	def print_verbose(self, tag):
		self._step_f = datetime.datetime.now()
		print("\ / {} - {}".format(tag, str(self._step_f - self._step_i)))
		self._step_i = self._step_f

	def print_verbose_end(self):
		self._step_f = datetime.datetime.now()
		self._end = self._step_f
		print(" V  END       - {} (TOOK {})".format(str(self._end), str(self._end - self._start)))


def setup():
	path_list = [ paths.DATA_PATH, paths.RAW_DATA_PATH, paths.TIER1_DATA_PATH, paths.TIER2_DATA_PATH, paths.BASELINE_DATA_PATH, paths.TRIALA_DATA_PATH, paths.TEMP_PATH, paths.RESULTS_PATH]

	for path in path_list:
		if not os.path.exists(path):
			os.makedirs(path)

	return


def dump_config(config_name):
	default_config = [
		["model_name", "ExampleFintech"],
		["ticker_list_samplesize", 100],
		["path_dataload_result", "RES_initial_dataload.csv"],
		["path_ticker_list", None],
		["date_from", '1900-01-01'],
		["date_to", str(datetime.date.today())],
		["fill_value", 1e-128],
		["ticker_list", None],
		["timespan", None],
		["timespan_ab", None],
		["train_from", "2010-01-01"],
		["train_until", "2015-12-31"],
		["test_from", "2016-01-01"]
		["exchanges", "nasdaq"],
		["feature_set", "baseline"]
	]

	config_path = "{}/{}.cfg".format(paths.CONFIG_PATH, config_name)

	pd.DataFrame(default_config, columns=["param", "value"]).to_csv(config_path, index=False, sep=":")

	return config_path



