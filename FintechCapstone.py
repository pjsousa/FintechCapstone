from __future__ import division, print_function

import math
import uuid
import datetime
import pandas as pd
import numpy as np
from functools import partial
from dateutil import parser as dtparser
from sklearn.externals import joblib

from utils import datafetch
from utils import datapipe
from utils import baseline_model
from utils import scenarioa
from utils import scenariob
from utils import paths_helper as paths

import argparse
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


TINY_FLOAT = 1e-128

class FinCapstone():

	def __init__(self,
				scenario="baseline",
				model_name="ExampleFintech",
				reset_status=False,
				ticker_list_samplesize=100,
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
				bins=None,):

		self.ticker_list = None
		self.timespan = None
		self.timespan_ab = None
		self.bins = None
		self.ticker_list_samplesize = ticker_list_samplesize
		self.date_from = date_from
		self.date_to = date_to
		self.fill_value = fill_value
		self.train_from = dtparser.parse(train_from)
		self.train_until = dtparser.parse(train_until)
		self.test_from = dtparser.parse(test_from)
		self.model_name = model_name
		self.scenario = scenario
		
		self.fetchstatus_df = None
		self.featureengineer_status_df = None
		self.train_status_df = None
		self.eval_status_df = None


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

		if reset_status:
			self.reset_status_files()
			self.store_status_files()
		else:
			self.load_status_files()



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
		ticker_list = None

		ticker_list = self.fetchstatus_df[self.fetchstatus_df["status"] == "OK"]
		ticker_list = ticker_list[~(self.featureengineer_status_df["status"] == "NOK")]

		return ticker_list



	def run_initial_dataload(self):
		_ok = None
		_nok = None
		_r = None

		# Fetch data for all tickers
		datafetch.initial_dataload(self.ticker_list, verbose=True, del_temp=True, status_df=self.fetchstatus_df)

		self.trialconfig_df.loc["fetch_status","value"] = "COMPLETE"
		self.store_status_files()



	def feature_engineering(self, scenario=None):
		scenario = self.scenario if scenario is None else scenario
		## Only work with tickers that were successfull during datafetch
		ticker_list = self.provision_validtickerlist()
		work_df = None
		iserror = None

		for ix, row in ticker_list.iterrows():
			try:
				_start = datetime.datetime.now()
				itr_ticker = ix
				print("\n\n - {} - \n".format(itr_ticker))
				itr_df = datafetch.load_raw_frame(itr_ticker,parseDate=False, dropAdjClose=True)
				
				itr_df = itr_df[pd.to_datetime(itr_df["Date"]) >= dtparser.parse(self.date_from)]
				itr_df = itr_df[pd.to_datetime(itr_df["Date"]) < dtparser.parse(self.date_to)]

				if scenario == "baseline":
					work_df = baseline_model.calc_features(itr_df,verbose=True)
					self.store_baseline_features(work_df, itr_ticker)

					work_df = baseline_model.calc_labels(itr_df, self.timespan, verbose=True)
					self.store_baseline_labels(work_df, itr_ticker)
				elif scenario == "scenarioa":
					itr_df.set_index("Date", inplace=True)
					work_df = scenarioa.calc_features(itr_df, normalize=True, verbose=True)
					self.store_scenarioa_features(work_df, itr_ticker)

					work_df = scenarioa.calc_labels(itr_df, verbose=True)
					self.store_scenarioa_labels(work_df, itr_ticker)
				elif scenario == "scenariob":
					itr_df.set_index("Date", inplace=True)
					work_df = scenariob.calc_features(itr_df, normalize=True, verbose=True)
					self.store_scenariob_features(work_df, itr_ticker)

					work_df = scenariob.calc_labels(itr_df, verbose=True)
					self.store_scenariob_labels(work_df, itr_ticker)
				else:
					print("Invalid Feature Set.")

				## KEEP track of status
				self.featureengineer_status_df.loc[itr_ticker, "status"] = "OK"
				self.featureengineer_status_df.loc[itr_ticker, "start"] = _start
				self.featureengineer_status_df.loc[itr_ticker, "end"] = datetime.datetime.now()
			except Exception as e:
				self.featureengineer_status_df.loc[itr_ticker, "status"] = "NOK"
				self.featureengineer_status_df.loc[itr_ticker, "start"] = _start
				self.featureengineer_status_df.loc[itr_ticker, "end"] = datetime.datetime.now()
				self.featureengineer_status_df.loc[itr_ticker, "msg"] = str(e)
				self.store_status_files()
				iserror = True

		self.trialconfig_df.loc["featureengineer_status","value"] = "ERRORS" if iserror else "COMPLETE"
		self.store_status_files()



	def split_tickerlist(self, n_splits=4):
		ticker_list = self.fetchstatus_df.index.tolist()
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



	def train(self, nb_epoch=1, train_next=None, ticker=None):
		## train only one specific ticker
		work_tickers = None
		_start = None
		iserror = False
		model = None

		if not(ticker is None):
			work_tickers = [ticker]
			train_next = 1
		else:
			work_tickers = self.train_status_df[self.train_status_df["status"] == "INCOMPLETE"].index.tolist()


		if (ticker is None) and (train_next is None):
			train_next = len(work_tickers)

		for idx_ticker in range(train_next):
			itr_ticker = work_tickers[idx_ticker]

			try:
				_start = datetime.datetime.now()

				if self.scenario == "baseline":
					model = self.train_baseline(itr_ticker, nb_epoch)
				elif self.scenario == "scenarioa":
					model = self.train_scenarioa(itr_ticker, nb_epoch)
				elif self.scenario == "scenariob":
					model = self.train_scenariob(None, nb_epoch)
				else:
					model = None


				self.train_status_df.loc[itr_ticker, "status"] = "OK"
				self.train_status_df.loc[itr_ticker, "epochs"] = nb_epoch + self.train_status_df.loc[itr_ticker, "epochs"]
				self.train_status_df.loc[itr_ticker, "start"] = _start
				self.train_status_df.loc[itr_ticker, "end"] = datetime.datetime.now()
				self.train_status_df.loc[itr_ticker, "msg"] = None
				self.store_status_files()

			except Exception as e:
				self.train_status_df.loc[itr_ticker, "status"] = "NOK"
				self.train_status_df.loc[itr_ticker, "epochs"] = self.train_status_df.loc[itr_ticker, "epochs"]
				self.train_status_df.loc[itr_ticker, "start"] = _start
				self.train_status_df.loc[itr_ticker, "end"] = datetime.datetime.now()
				self.train_status_df.loc[itr_ticker, "msg"] = str(e)
				self.store_status_files()
				iserror = True

		work_tickers = self.train_status_df[self.train_status_df["status"] == "INCOMPLETE"].index.tolist()

		if len(work_tickers) == 0:
			self.trialconfig_df.loc["modeltrain_status", "value"] = "ERRORS" if iserror else "DONE"

		self.store_status_files()


	def evaluate(self, train_next=None, ticker=None):
		work_tickers = None
		_start = None
		iserror = False
		model = None
		evals = None

		if not(ticker is None):
			work_tickers = [ticker]
			train_next = 1
		else:
			work_tickers = self.eval_status_df[self.eval_status_df["status"] == "INCOMPLETE"].index.tolist()


		if (ticker is None) and (train_next is None):
			train_next = len(work_tickers)


		if self.scenario == "baseline":
			print("EVAL BA model for %s tickers" % n_tickers)
			model = baseline_model.create_model()
		elif self.scenario == "scenarioa":
			n_tickers = len(self.valid_ticker_list())
			print("EVAL SA model for %s tickers" % n_tickers)
			model = scenarioa.create_model(n_tickers)
		elif self.scenario == "scenariob":
			n_tickers = len(self.valid_ticker_list())
			print("EVAL SA model for %s tickers" % n_tickers)
			model = scenariob.create_model(n_tickers)


		for idx_ticker in range(train_next):
			itr_ticker = work_tickers[idx_ticker]
			try:
				_start = datetime.datetime.now()

				if self.scenario == "baseline":
					evals = self.evaluate_baseline(itr_ticker, model)
				elif self.scenario == "scenarioa":
					evals = self.evaluate_scenarioa(itr_ticker, model)
				elif self.scenario == "scenariob":
					evals = self.evaluate_scenariob(itr_ticker, model)

				self.eval_status_df.loc[itr_ticker, "status"] = "OK"
				self.eval_status_df.loc[itr_ticker, "epochs"] = self.train_status_df.loc[itr_ticker, "epochs"]
				self.eval_status_df.loc[itr_ticker, "start"] = _start
				self.eval_status_df.loc[itr_ticker, "end"] = datetime.datetime.now()
				self.eval_status_df.loc[itr_ticker, "r_squared"] = evals["r_squared"]
				self.eval_status_df.loc[itr_ticker, "accuracy"] = evals["accuracy"]
			except Exception as e:
				self.eval_status_df.loc[itr_ticker, "status"] = "NOK"
				self.eval_status_df.loc[itr_ticker, "epochs"] = None
				self.eval_status_df.loc[itr_ticker, "start"] = _start
				self.eval_status_df.loc[itr_ticker, "end"] = datetime.datetime.now()
				self.eval_status_df.loc[itr_ticker, "r_squared"] = None
				self.eval_status_df.loc[itr_ticker, "accuracy"] = None
				self.eval_status_df.loc[itr_ticker, "msg"] = str(e)

				self.store_status_files()
				iserror = True

		if len(work_tickers) == 0:
			self.trialconfig_df.loc["modeleval_status", "value"] = "ERRORS" if iserror else "DONE"

		self.store_status_files()


	def valid_ticker_list(self):
		_r = self.provision_validtickerlist().index.tolist()

		return _r


	def train_baseline(self, ticker, nb_epoch=100):
		results = None
		X_train = None
		y_train = None
		X_test = None
		y_test = None

		print("Training Baseline for {}, {}".format(ticker, nb_epoch))

		features_df = self.load_baseline_features(ticker, parseDate=True)
		features_df.set_index("Date", inplace=True)

		labels_df = self.load_baseline_labels(ticker, parseDate=True)
		labels_df.set_index("Date", inplace=True)

		model = baseline_model.create_model()
		X_train, y_train, X_test, y_test = baseline_model.prepare_problemspace(features_df, labels_df, self.train_from, self.train_until, self.test_from)

		for step_idx in np.arange(nb_epoch / 50):
			baseline_model.fit(model, X_train, y_train, 50)

			model.save_weights("{}/weights{}_{}_{}_step{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, ticker, step_idx))
			model.save_weights("{}/weights{}_{}_{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, ticker))

			results = self.evaluate_baseline(ticker, model, (X_train, y_train, X_test, y_test))

			print(results)

		

		return model


	def evaluate_baseline(self, ticker, model=None, data=None):
		_r = None
		X_train = None
		y_train = None
		X_test = None
		y_test = None

		if model is None:
			model = baseline_model.create_model()
			model.load_weights("{}/weights{}_{}_{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, ticker))

		if data is None:
			features = self.load_baseline_features(ticker, True).set_index("Date")
			labels = self.load_baseline_labels(ticker, True).set_index("Date")

			X_train, y_train, X_test, y_test = baseline_model.prepare_problemspace(features, labels, self.train_from, self.train_until, self.test_from, "numpy")
		else:
			X_train = data[0]
			y_train = data[1]
			X_test = data[2]
			y_test = data[3]

		print("Evaluating {}".format(ticker))
		_r = baseline_model.evaluate(model, X_test, y_test, return_type="dict")

		return _r


	def train_scenarioa(self, ticker, nb_epoch=100):
		X_train = None
		y_train = None
		X_test = None
		y_test = None
		n_tickers = None
		results = None

		print("Training ScenarioA for {}, {}".format(ticker, nb_epoch))

		n_tickers  = len(self.valid_ticker_list())

		print("TRAIN model for %s tickers" % n_tickers)

		model = scenarioa.create_model(n_tickers)
		X_train, y_train, X_test, y_test = scenarioa.prepare_problemspace(ticker, self.valid_ticker_list(), self.train_from, self.train_until, self.test_from, True, "numpy")
		
		for step_idx in np.arange(nb_epoch / 10):
			scenarioa.fit(model, X_train, y_train, nb_epoch=10)
			model.save_weights("{}/weights{}_{}_{}_step{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, ticker, step_idx))
			model.save_weights("{}/weights{}_{}_{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, ticker))

			results = self.evaluate_scenarioa(ticker, model, (X_train, y_train, X_test, y_test))
			print(results)


		return model



	def evaluate_scenarioa(self, ticker, model=None, data=None):
		_r = None
		X_train = None
		y_train = None
		X_test = None
		y_test = None

		if model is None:
			n_tickers = len(self.valid_ticker_list())
			model = scenarioa.create_model(n_tickers)
			model.load_weights("{}/weights{}_{}_{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, ticker))

		if data is None:
			features = self.load_baseline_features(ticker, True).set_index("Date")
			labels = self.load_baseline_labels(ticker, True).set_index("Date")

			X_train, y_train, X_test, y_test = scenarioa.prepare_problemspace(ticker, self.valid_ticker_list(), self.train_from, self.train_until, self.test_from, True, "numpy")
		else:
			X_train = data[0]
			y_train = data[1]
			X_test = data[2]
			y_test = data[3]

		print("Evaluating {}".format(ticker))
		_start = datetime.datetime.now()
		_r = scenarioa.evaluate(model, X_test, y_test, return_type="dict")

		return _r



	def train_scenariob(self, ticker=None, nb_epoch=100):
		X_train = None
		y_train = None
		X_test = None
		y_test = None
		n_tickers = None
		results = None


		if ticker is None:
			X_train, y_train, X_test, y_test = scenariob.prepare_problemspace(self.valid_ticker_list(), self.train_from, self.train_until, self.test_from, normalize=True, return_type="numpy")
			X_final, pca = scenariob.dim_reduction(X_train, 900)

			model = scenariob.create_model(len(self.valid_ticker_list()), X_final.shape[1])
			joblib.dump( pca, "{}/pca_{}_{}_{}.p".format(paths.TEMP_PATH, self.scenario, self.model_name, "MARKET"))


			for step_idx in np.arange(nb_epoch / 50):
				scenariob.fit(model, X_train, y_train, nb_epoch=50)
				model.save_weights("{}/weights{}_{}_{}_step{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, ticker, step_idx))
				model.save_weights("{}/weights{}_{}_{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, ticker))

				results = self.evaluate_scenariob(ticker, model, (X_train, y_train, X_test, y_test), pca)
				print(results)
				
		else:
			assert 1 > 2, \
				"Not implemented yet"

		return model


	def evaluate_scenariob(self, ticker, model=None, data=None, pca=None):
		_r = None
		X_train = None
		y_train = None
		X_test = None
		y_test = None


		if (data is None) :
			X_train, y_train, X_test, y_test = scenariob.prepare_problemspace(self.valid_ticker_list(), self.train_from, self.train_until, self.test_from, normalize=True, return_type="numpy")
		else:
			X_train = data[0]
			y_train = data[1]
			X_test = data[2]
			y_test = data[3]
		
		if pca is None:
			pca = joblib.load("{}/pca_{}_{}_{}.p".format(paths.TEMP_PATH, self.scenario, self.model_name, "MARKET"))


		if model is None:
			X_final, pca = scenariob.dim_reduction(X_train, 900, pca)
			model = scenariob.create_model(len(self.valid_ticker_list()), X_final.shape[1])
			model.load_weights("{}/weights{}_{}_{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, ticker))
		
		print("Evaluating {}".format(ticker))
		_start = datetime.datetime.now()
		y_pred = model.predict(X_test, verbose=0)
		_r = scenariob.evaluate(model, X_test, y_test, return_type="dict")

		return _r


	##################
	## Status Files ##
	##################
	def reset_status_files(self):
		data = {
			"date_from": self.date_from
			,"date_to": self.date_to
			,"train_from": self.train_from
			,"train_until": self.train_until
			,"test_from": self.test_from
			,"model_name": self.model_name
			,"scenario": self.scenario
			,"fetch_status": "INCOMPLETE"
			,"featureengineer_status": "INCOMPLETE"
			,"modeltrain_status": "INCOMPLETE"
			,"modeleval_status": "INCOMPLETE"
		}

		trialconfig_df = pd.DataFrame.from_dict(data, orient='index')
		trialconfig_df.index.name = "property"
		trialconfig_df.columns = pd.Index(["value"])


		fetchstatus_df = pd.DataFrame()
		fetchstatus_df["ticker"] = self.ticker_list
		fetchstatus_df["status"] = "INCOMPLETE"
		fetchstatus_df["start"] = None
		fetchstatus_df["end"] = None
		fetchstatus_df["msg"] = None
		fetchstatus_df.set_index("ticker", inplace=True)


		featureengineer_status_df = pd.DataFrame()
		featureengineer_status_df["ticker"] = self.ticker_list
		featureengineer_status_df["status"] = "INCOMPLETE"
		featureengineer_status_df["start"] = None
		featureengineer_status_df["end"] = None
		featureengineer_status_df["msg"] = None
		featureengineer_status_df.set_index("ticker", inplace=True)


		train_status_df = pd.DataFrame()
		train_status_df["ticker"] = self.ticker_list
		train_status_df["epochs"] = 0
		train_status_df["status"] = "INCOMPLETE"
		train_status_df["start"] = None
		train_status_df["end"] = None
		train_status_df["msg"] = None
		train_status_df["loss"] = None
		train_status_df.set_index("ticker", inplace=True)

		eval_status_df = pd.DataFrame()
		eval_status_df["ticker"] = self.ticker_list
		eval_status_df["epochs"] = 0
		eval_status_df["status"] = "INCOMPLETE"
		eval_status_df["start"] = None
		eval_status_df["end"] = None
		eval_status_df["r_squared"] = None
		eval_status_df["accuracy"] = None
		eval_status_df["msg"] = None
		eval_status_df.set_index("ticker", inplace=True)

		self.trialconfig_df = trialconfig_df
		self.fetchstatus_df = fetchstatus_df
		self.featureengineer_status_df = featureengineer_status_df
		self.train_status_df = train_status_df
		self.eval_status_df = eval_status_df

	def store_status_files(self):
		self.trialconfig_df.to_csv("{}_trialconfig.tmp".format(self.model_name))
		self.fetchstatus_df.to_csv("{}_fetchstatus.tmp".format(self.model_name))
		self.featureengineer_status_df.to_csv("{}_featureengineer_status.tmp".format(self.model_name))
		self.train_status_df.to_csv("{}_train_status.tmp".format(self.model_name))
		self.eval_status_df.to_csv("{}_eval_status.tmp".format(self.model_name))

	def load_status_files(self):
		self.trialconfig_df = pd.read_csv("{}_trialconfig.tmp".format(self.model_name))
		self.fetchstatus_df = pd.read_csv("{}_fetchstatus.tmp".format(self.model_name))
		self.featureengineer_status_df = pd.read_csv("{}_featureengineer_status.tmp".format(self.model_name))
		self.train_status_df = pd.read_csv("{}_train_status.tmp".format(self.model_name))
		self.eval_status_df = pd.read_csv("{}_eval_status.tmp".format(self.model_name))

		self.trialconfig_df.set_index("property", inplace=True)
		self.fetchstatus_df.set_index("ticker", inplace=True)
		self.featureengineer_status_df.set_index("ticker", inplace=True)
		self.train_status_df.set_index("ticker", inplace=True)
		self.eval_status_df.set_index("ticker", inplace=True)

	#####################
	## Storage Helpers ##
	#####################
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
		
		return scenarioa.load_scenarioa_features(ticker, parseDate)

	def store_scenarioa_labels(self, features_df, ticker):
		
		return scenarioa.store_scenarioa_labels(features_df, ticker)

	def load_scenarioa_labels(self, ticker, parseDate=True):
		
		return scenarioa.load_scenarioa_labels(ticker, parseDate)


	def store_scenariob_features(self, features_df, ticker):
		
		return scenariob.store_scenariob_features(features_df, ticker)

	def load_scenariob_features(self, ticker, parseDate=True):
		
		return scenariob.load_scenariob_features(ticker, parseDate)

	def store_scenariob_labels(self, features_df, ticker):
		
		return scenariob.store_scenariob_labels(features_df, ticker)

	def load_scenariob_labels(self, ticker, parseDate=True):
		
		return scenariob.load_scenariob_labels(ticker, parseDate)


	#####################
	## Verbose Helpers ##
	#####################
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
		["scenario", "baseline"]
	]

	config_path = "{}/{}.cfg".format(paths.CONFIG_PATH, config_name)

	pd.DataFrame(default_config, columns=["param", "value"]).to_csv(config_path, index=False, sep=":")

	return config_path



