from __future__ import division, print_function

import math
import uuid
import datetime
import pandas as pd
import numpy as np
import itertools
from functools import partial
from dateutil import parser as dtparser
from sklearn.externals import joblib


from utils import datafetch
from utils.datafetch import print_progress
from utils import datapipe
from utils import baseline_model
from utils import scenarioa
from utils import scenariob
from utils import scenarioc
from utils import paths_helper as paths

import argparse
import os
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TINY_FLOAT = 1e-10

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
				train_from="2010-02-15",
				train_until="2014-12-31",
				test_from="2016-02-15",
				test_until="2016-12-31",
				encode_workpages=7,
				bins=None):

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
		self.test_until = dtparser.parse(test_until)
		self.model_name = model_name
		self.scenario = scenario
		self.encode_workpages = encode_workpages
		
		self.trialconfig_df = None
		self.fetchstatus_df = None
		self.featureengineer_status_df = None
		self.train_status_df = None
		self.eval_status_df = None
		self.encode_status_df = None

		for i in range(encode_workpages):
			self.__dict__["encode_status_df_%d" % i] = None


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
			for i in range(encode_workpages):
				self.store_encodestatus_files(i)
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

		return ticker_list.index.tolist()



	def run_initial_dataload(self):
		_ok = None
		_nok = None
		_r = None

		# Fetch data for all tickers
		datafetch.initial_dataload(self.ticker_list, verbose=True, del_temp=True, status_df=self.fetchstatus_df)

		self.trialconfig_df.loc["fetch_status","value"] = "COMPLETE"
		self.store_status_files()



	def feature_engineering(self, scenario=None, ticker_list=None):
		"""
		Based off of the raw data, performs the feature and label calculations for the specified scenario.
		Tracks time and status of the process for each ticker.
		"""
		scenario = self.scenario if scenario is None else scenario
		
		## Only work with tickers that were successfull during datafetch
		ticker_list = self.provision_validtickerlist() if ticker_list is None else ticker_list
		ticker_count = len(ticker_list)
		itr_count = 0
		work_df = None
		iserror = None

		print("\n{} - Feature/Label Engineering - {} - [{} ; {}] ".format(datetime.datetime.now(), scenario, self.date_from, self.date_to))

		for itr_ticker in ticker_list:
			itr_count += 1
			try:
				_start = datetime.datetime.now()

				## Load raw data and slice for the dates we want for our problem
				itr_df = datafetch.load_raw_frame(itr_ticker,parseDate=False, dropAdjClose=False)
				
				itr_df = itr_df[pd.to_datetime(itr_df["Date"]) >= dtparser.parse(self.date_from)]
				itr_df = itr_df[pd.to_datetime(itr_df["Date"]) < dtparser.parse(self.date_to)]

				## Calculate depending on scenario and store
				if scenario == "baseline":
					print_progress("  ({}/{}) - Calculating Features {}".format(itr_count, ticker_count, itr_ticker))
					work_df = baseline_model.calc_features(itr_df,verbose=True)
					self.store_baseline_features(work_df, itr_ticker)

					print_progress("  ({}/{}) - Calculating Labels {}".format(itr_count, ticker_count, itr_ticker))
					work_df = baseline_model.calc_labels(itr_df, self.timespan, verbose=True)
					self.store_baseline_labels(work_df, itr_ticker)
				elif scenario == "scenarioc":
					print_progress("  ({}/{}) - Calculating Features {}".format(itr_count, ticker_count, itr_ticker))
					itr_df.set_index("Date", inplace=True)
					work_df = scenarioc.calc_features(itr_df, verbose=True)
					self.store_scenarioc_features(work_df, itr_ticker)

					print_progress("  ({}/{}) - Calculating Labels {}".format(itr_count, ticker_count, itr_ticker))
					work_df = scenarioc.calc_labels(itr_df, verbose=True)
					self.store_scenarioc_labels(work_df, itr_ticker)
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

		print("\n{} - Feature/Label Engineering Finished".format(datetime.datetime.now()))

		self.trialconfig_df.loc["featureengineer_status","value"] = "ERRORS" if iserror else "COMPLETE"
		self.store_status_files()



	def feature_encoding(self, scenario=None, work_page=None, useSample=None, timespan=224, bins=100):
		"""
		Runs the encoding process used for scenarioc.
		The resulting encoded images will have the dimensions [timespan x timespan x number of features] and 
		will be discretized to an arbitrary 'bins' size.
		We can run the encoding only on a subsample of the problem space.
		The whole problem space is paginated so we can also choose the page to run (so we can run all the pages in pararel)
		"""
		scenario = self.scenario if scenario is None else scenario
		_skip = False
		_skip_count = 0
		_done_count = 0
		_err_count = 0

		## Settle the whole working problem space
		number_pages = self.encode_workpages
		_status_df = self.encode_status_df
		
		tickers = self.valid_ticker_list()
		dates = [ datetime.datetime.strftime(x, "%Y-%m-%d") for x in pd.date_range(start=self.date_from, end=self.date_to)]

		_todo = [tickers, dates, [bins], [timespan], [self.model_name]]
		_todo = [x for x in itertools.product(*_todo)]

		# ## Settle page to run. if work_page is defined, slice by the given page
		if not(work_page is None):
			page_size = int(np.ceil(len(_todo) / number_pages))
			_todo = _todo[page_size*work_page:page_size*(work_page + 1)]
			_status_df = self.__dict__["encode_status_df_%d" % work_page]

		itr_keys = ["itr_ticker", "itr_date", "n_states", "timespans"]

		print("\n{} - F. Encoding - {} - Page {} - Timespan {} - N_BINS {}".format(datetime.datetime.now(), scenario, work_page, timespan, bins))
		print("{} Tickers - {} Dates - Sampling {} observations with {} change".format(len(tickers), len(dates),len(_todo), useSample))

		for itr in _todo:
			_start = datetime.datetime.now()

			try:
				## skips the iteration if the random is above our subsampling change
				if (useSample is not None) and (np.random.rand() > useSample):
					#print("Skip. {} {}".format(itr[0], itr[1]))
					_skip_count += 1
					continue

				## skips if we already created the image some other run
				if scenarioc.check_encoding_exists(itr[0], itr[1], itr[3], itr[2]):
					#print("File Exists. {} {}".format(itr[0], itr[1]))
					_skip_count += 1
					continue

				## call encode
				mtf = scenarioc.encode_features(*itr)

				self.store_scenarioc_encodings(mtf, itr[0], itr[1], itr[3], itr[2])
				_done_count += 1
				print_progress("  Last encoding {} {} [{} Done] [{} Skipped] [{} Weekend/Holiday]".format(itr[0], itr[1], _done_count, _skip_count, _err_count))
				
				_status_df.loc[(itr[0], itr[1]), "start"] = _start
				_status_df.loc[(itr[0], itr[1]), "end"] = datetime.datetime.now()
				_status_df.loc[(itr[0], itr[1]), "status"] = "OK"

			except KeyError:
				_err_count += 1
				#print("KeyError {}.".format(itr))
			except Exception as e:
				_status_df.loc[(itr[0], itr[1]), "start"] = _start
				_status_df.loc[(itr[0], itr[1]), "end"] = datetime.datetime.now()
				_status_df.loc[(itr[0], itr[1]), "status"] = "NOK"
				_status_df.loc[(itr[0], itr[1]), "msg"] = str(e)
				_err_count += 1

			self.store_encodestatus_files(work_page)

		print_progress("  Last encoding {} {} [{} Done] [{} Skipped] [{} Weekend/Holiday]".format(itr[0], itr[1], _done_count, _skip_count, _err_count))
		print("\n{} - F. Encoding Finished- Page {}".format(datetime.datetime.now(), work_page))



	def split_tickerlist(self, n_splits=4):
		"""
		Deprecate?
		"""
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



	def train(self, nb_epoch=1, train_next=None, ticker=None, useSample=None, input_shape=(224,224,3), filter_shape=(3, 3), output_size=3, FC_layers=4, earlystop=5, timespan=224, bins=100, finetune=None, dropout=0.0, optimizer="adam"):
		"""
		Train session wrapper.
		Starts the training session for the specified scenario.
		"""

		work_tickers = None
		_start = None
		iserror = False
		model = None

		print("\n{} - Train Session - {} - {} epochs - Ticker {}".format(datetime.datetime.now(), self.scenario, nb_epoch, ticker))

		## The train is done Ticker by Ticker in the case of the baseline scenario. We are creating multiple models in that scenario.
		if not(ticker is None):
			work_tickers = [ticker]
			train_next = 1
		else:
			work_tickers = self.train_status_df[self.train_status_df["status"] == "INCOMPLETE"].index.tolist()


		if (ticker is None) and (train_next is None):
			train_next = len(work_tickers)

		for idx_ticker in range(train_next):
			itr_ticker = work_tickers[idx_ticker]

			## The specifif training session
			try:
				_start = datetime.datetime.now()

				if self.scenario == "baseline":
					model = self.train_baseline(itr_ticker, nb_epoch)
				elif self.scenario == "scenarioc":
					
					model = self.train_scenarioc(nb_epoch, useSample, input_shape=input_shape, filter_shape=filter_shape, output_size=output_size
											, FC_layers=FC_layers, earlystop=earlystop, timespan=timespan, bins=bins, finetune=finetune
											, dropout=dropout, optimizer=optimizer)
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
				print(str(e))

		work_tickers = self.train_status_df[self.train_status_df["status"] == "INCOMPLETE"].index.tolist()

		if len(work_tickers) == 0:
			self.trialconfig_df.loc["modeltrain_status", "value"] = "ERRORS" if iserror else "DONE"

		self.store_status_files()


	def evaluate(self, train_next=None, ticker=None, input_shape=(224,224,3), filter_shape=(3, 3), output_size=3, FC_layers=4, timespan=224, bins=100, finetune_path=None, dropout=0.0, optimizer="adam"):
		"""
		Loads and evaluates a model against our test data.
		"""
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
		elif self.scenario == "scenarioc":
			n_tickers = len(self.valid_ticker_list())
			print("EVAL SA model for %s tickers" % n_tickers)
			model = scenariob.create_model(input_shape=input_shape, filter_shape=filter_shape, output_size=output_size, FC_layers=FC_layers, timespan=timespan, bins=bins, finetune_path=finetune_path, dropout=dropout, optimizer=optimizer)

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
		_r = self.provision_validtickerlist()

		return _r


	def train_baseline(self, ticker, nb_epoch=100):
		"""
		Runs a training session for the baseline model
		"""
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

		for step_idx in np.arange(nb_epoch / 2):
			_start = datetime.datetime.now()
			_epoch_index = int(((step_idx*2)+2))

			baseline_model.fit(model, X_train, y_train, 2)

			model.save_weights("{}/weights{}_{}_{}_step{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, ticker, _epoch_index))
			model.save_weights("{}/weights{}_{}_{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, ticker))

			results = self.evaluate_baseline(ticker, model, (X_train, y_train, X_test, y_test))

			print(results)
			
			self.eval_status_df.loc[(ticker, _epoch_index), "status"] = "COMPLETE"
			self.eval_status_df.loc[(ticker, _epoch_index), "start"] = _start
			self.eval_status_df.loc[(ticker, _epoch_index), "end"] = datetime.datetime.now()
			self.eval_status_df.loc[(ticker, _epoch_index), "r_squared"] = results[0]["r_squared"]
			self.eval_status_df.loc[(ticker, _epoch_index), "accuracy"] = results[0]["accuracy"]
			self.eval_status_df.loc[(ticker, _epoch_index), "r_squared_test"] = results[1]["r_squared"]
			self.eval_status_df.loc[(ticker, _epoch_index), "accuracy_test"] = results[1]["accuracy"]

			self.store_status_files()

		return model

	def evaluate_baseline(self, ticker, model=None, data=None):
		_r = [None] * 2
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
		_r[0] = baseline_model.evaluate(model, X_train, y_train, return_type="dict")
		_r[1] = baseline_model.evaluate(model, X_test, y_test, return_type="dict")

		return _r

	def train_scenarioc(self, nb_epoch=100, useSample=None, input_shape=(224,224,3), filter_shape=(3, 3), output_size=3, FC_layers=4, earlystop=5, timespan=224, bins=100, finetune=None, dropout=0.0, optimizer="adam"):
		model = None
		best_epoch = None
		train_eval = pd.DataFrame(columns=["mse", "r_squared", "accuracy"])
		valid_eval = pd.DataFrame(columns=["mse", "r_squared", "accuracy"])

		print("Training Scenario C")
		print("train_from={}, train_until={}, test_from={}, test_until={}".format(datetime.datetime.strftime(self.train_from, "%Y-%m-%d"), datetime.datetime.strftime(self.train_until, "%Y-%m-%d"), datetime.datetime.strftime(self.test_from, "%Y-%m-%d"), datetime.datetime.strftime(self.test_until, "%Y-%m-%d")))
		print("input_shape={}, bins={}, filter_shape={}, output_size={}, FC_layers={}".format(input_shape, bins, filter_shape, output_size, FC_layers))
		print("sample={}, earlystop={}".format(useSample, earlystop))
		print("\n")

		## load all label data and feature contexts for batch loading
		_tickers, _dates, _labels = scenarioc.prepare_problemspace(self.valid_ticker_list(), timespan, bins)

		if useSample:
			_tickers = _tickers[:int(_tickers.shape[0] * useSample)]
			_dates = _dates[:int(_dates.shape[0] * useSample)]
		else:
			useSample = 1.0

		## Lets slice out the "TEST"
		_mask_train = (_dates > pd.to_datetime(self.train_from)) & (_dates < pd.to_datetime(self.train_until)) 
		#_mask_test = (_dates >= pd.to_datetime(self.test_from))
		_tickers = _tickers[_mask_train]
		_dates = _dates[_mask_train]

		## And Split train into train and validation with an 80% 20% split
		_mask_trainvalid = np.arange(_tickers.shape[0])
		np.random.shuffle(_mask_trainvalid)
		_tickers_train = _tickers[_mask_trainvalid[int(np.ceil(_mask_trainvalid.shape[0] * 0.2)):]]
		_dates_train = _dates[_mask_trainvalid[int(np.ceil(_mask_trainvalid.shape[0] * 0.2)):]]
		_tickers_valid = _tickers[_mask_trainvalid[:int(np.ceil(_mask_trainvalid.shape[0] * 0.2))]]
		_dates_valid = _dates[_mask_trainvalid[:int(np.ceil(_mask_trainvalid.shape[0] * 0.2))]]

		print("Shapes: [TICKER {}T {}V] [DATES {}T {}V]".format(_tickers_train.shape[0], _tickers_valid.shape[0], _dates_train.shape[0], _dates_valid.shape[0]))
		

		model = scenarioc.create_model(input_shape, filter_shape, output_size, FC_layers)

		if finetune is not None:
			model.load_weights("{}/weights{}_{}.h5".format(paths.TEMP_PATH, self.scenario, finetune))
			scenarioc.finetune(model, output_size=output_size, FC_layers=FC_layers, dropout=dropout, optimizer=optimizer)

		feature_mean, feature_std = scenarioc.features_stats(_dates_train, _tickers_train, _labels, timespan, bins)

		for itr_epoch in range(nb_epoch):
			_start = datetime.datetime.now()
			print_progress("  Epoch {} - TRAINING ".format(itr_epoch))

			scenarioc.train(model, _dates_train, _tickers_train, _labels, timespan, bins, feature_mean, feature_std)

			print_progress("  Epoch {} - EVAL. TRAIN ".format(itr_epoch))
			train_eval = scenarioc.evaluate(model, _dates_train, _tickers_train, _labels, timespan, bins, feature_mean, feature_std)
			print_progress("  Epoch {} - EVAL. VALID ".format(itr_epoch))
			valid_eval = scenarioc.evaluate(model, _dates_valid, _tickers_valid, _labels, timespan, bins, feature_mean, feature_std)

			self.eval_status_df.loc[("Nan", itr_epoch), "status"] = "COMPLETE"
			self.eval_status_df.loc[("Nan", itr_epoch), "start"] = _start
			self.eval_status_df.loc[("Nan", itr_epoch), "end"] = datetime.datetime.now()
			self.eval_status_df.loc[("Nan", itr_epoch), "mse"] = train_eval["mse"]
			self.eval_status_df.loc[("Nan", itr_epoch), "r_squared"] = train_eval["r_squared"]
			self.eval_status_df.loc[("Nan", itr_epoch), "accuracy"] = train_eval["accuracy"]
			self.eval_status_df.loc[("Nan", itr_epoch), "mse_test"] = valid_eval["mse"]
			self.eval_status_df.loc[("Nan", itr_epoch), "r_squared_test"] = valid_eval["r_squared"]
			self.eval_status_df.loc[("Nan", itr_epoch), "accuracy_test"] = valid_eval["accuracy"]

			self.store_status_files()


			if best_epoch is None:
				best_epoch = [itr_epoch, valid_eval["r_squared"]]
				print_progress("  Epoch {} - DUMP WEIGTHS ".format(itr_epoch))
				model.save_weights("{}/weights{}_{}_{}_step{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, useSample, itr_epoch))
				model.save_weights("{}/weights{}_{}_{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, useSample))
			else:
				if valid_eval["r_squared"] > best_epoch[1]:
					best_epoch = [itr_epoch, valid_eval["r_squared"]]

					print_progress("  Epoch {} - DUMP WEIGTHS ".format(itr_epoch))
					model.save_weights("{}/weights{}_{}_{}_step{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, useSample, itr_epoch))
					model.save_weights("{}/weights{}_{}_{}.h5".format(paths.TEMP_PATH, self.scenario, self.model_name, useSample))


				if (itr_epoch - best_epoch[0]) > earlystop:
					print("Not improving for %s epochs. Stopping." % earlystop)
					break;
			print_progress("  Epoch {} - [M, R, A] - T[{:.6f},{:.6f},{:.6f}] V[{:.6f},{:.6f},{:.6f}] {} ".format(itr_epoch, train_eval["mse"], train_eval["r_squared"], train_eval["accuracy"], valid_eval["mse"], valid_eval["r_squared"], valid_eval["accuracy"], ("*" if itr_epoch - best_epoch[0] == 0 else "")))
			print("\n")

		return model

	def evaluate_scenarioc(self,finetune_path, input_shape=(224,224,3), filter_shape=(3, 3), output_size=3, FC_layers=4, timespan=224, bins=100, dropout=0.0, optimizer="adam"):
		model = None
		train_eval = pd.DataFrame(columns=["mse", "r_squared", "accuracy"])
		test_eval = pd.DataFrame(columns=["mse", "r_squared", "accuracy"])
		_tickers = None
		_dates = None
		_labels = None
		_mask_train = None
		_mask_test = None
		_tickers_train = None
		_dates_train = None
		_tickers_test = None
		_dates_test = None

		print("Training Scenario C")
		print("train_from={}, train_until={}, test_from={}, test_until={}".format(datetime.datetime.strftime(self.train_from, "%Y-%m-%d"), datetime.datetime.strftime(self.train_until, "%Y-%m-%d"), datetime.datetime.strftime(self.test_from, "%Y-%m-%d"), datetime.datetime.strftime(self.test_until, "%Y-%m-%d")))
		print("input_shape={}, bins={}, filter_shape={}, output_size={}, FC_layers={}".format(input_shape, bins, filter_shape, output_size, FC_layers))
		print("\n")

		## load all label data and feature contexts for batch loading
		_tickers, _dates, _labels = scenarioc.prepare_problemspace(self.valid_ticker_list(), timespan, bins)

		## We'll still want the train data. We'll use it to find the mean and std. deviation of the data.
		_mask_train = (_dates > pd.to_datetime(self.train_from)) & (_dates < pd.to_datetime(self.train_until)) 
		_mask_test = (_dates >= pd.to_datetime(self.test_from))
		
		_tickers_train = _tickers[_mask_train]
		_dates_train = _dates[_mask_train]
		_tickers_test = _tickers[_mask_test]
		_dates_test = _dates[_mask_test]
		print("Shapes: [TICKER {}Tst] [DATES {}Tst]".format(_tickers_test.shape[0], _dates_test.shape[0]))

		print_progress("\n Creating model and loading weights")
		model = scenarioc.create_model(input_shape, filter_shape, output_size, FC_layers)
		model.load_weights("{}/weights{}_{}.h5".format(paths.TEMP_PATH, self.scenario, finetune_path))

		print_progress("\n Finding Mean and Std. Deviation")
		feature_mean, feature_std = scenarioc.features_stats(_dates_train, _tickers_train, _labels, timespan, bins)

		_start = datetime.datetime.now()
		print_progress("  Epoch {} - EVAL. TEST ".format(-1))
		test_eval = scenarioc.evaluate(model, _dates_test, _tickers_test, _labels, timespan, bins, feature_mean, feature_std)

		self.eval_status_df.loc[("Nan", -1), "status"] = "COMPLETE"
		self.eval_status_df.loc[("Nan", -1), "start"] = _start
		self.eval_status_df.loc[("Nan", -1), "end"] = datetime.datetime.now()
		self.eval_status_df.loc[("Nan", -1), "mse"] = train_eval["mse"]
		self.eval_status_df.loc[("Nan", -1), "r_squared"] = train_eval["r_squared"]
		self.eval_status_df.loc[("Nan", -1), "accuracy"] = train_eval["accuracy"]
		self.eval_status_df.loc[("Nan", -1), "mse_test"] = test_eval["mse"]
		self.eval_status_df.loc[("Nan", -1), "r_squared_test"] = test_eval["r_squared"]
		self.eval_status_df.loc[("Nan", -1), "accuracy_test"] = test_eval["accuracy"]

		self.store_status_files()

		print_progress("  Epoch {} - [M, R, A] - T[{:.6f},{:.6f},{:.6f}]".format(-1, test_eval["mse"], test_eval["r_squared"], test_eval["accuracy"]))
		print("\n")

		return model

	def predict_scenarioc(self,finetune_path, tickers, dates, input_shape=(224,224,3), filter_shape=(3, 3), output_size=3, FC_layers=4, timespan=224, bins=100, dropout=0.0, optimizer="adam"):
		print_progress("Finding distinct tickers")
		time.sleep(1)
		print_progress("Force-Fetching necessary tickers")
		time.sleep(1)
		print_progress("Running Feature Engineering")
		time.sleep(1)
		print_progress("Encoding missing images")
		time.sleep(1)
		print_progress("Running Predict")
		time.sleep(1)


		_tickers_predict = ["NVDA","NFLX", "NVDA"]
		_dates_predict = ["2017-03-03", "2017-03-03", "2017-03-06"]

		input_shape=(224,224,3)
		filter_shape=(3, 3)
		output_size=3
		FC_layers=4
		timespan=224
		bins=100
		finetune_path=None
		dropout=0.0
		optimizer="adam"


		_contexts = [tuple(x) for x in zip(_tickers_predict, pd.to_datetime(_dates_predict))]
		_distinct_tickers = np.unique(_tickers_predict)
		_distinct_tickers.tolist()
		_distinct_dates = pd.to_datetime(np.unique(_dates_predict))
		_distinct_dates.tolist()

		print_progress("Force-Fetching necessary tickers\n")
		initial_dataload(_distinct_tickers, force=True)

		print_progress("Running Feature Engineering")
		trial.feature_engineering(ticker_list=_distinct_tickers.tolist())


		_skip_count = 0
		_done_count = 0
		_err_count = 0

		print_progress("Encoding missing images")
		for itr_ticker, itr_date in zip(_tickers_predict, _dates_predict):
			## skips if we already created the image some other run
			if scenarioc.check_encoding_exists(itr_ticker, itr_date, timespan, bins):
				#print("File Exists. {} {}".format(itr[0], itr[1]))
				_skip_count += 1
				continue

				## call encode
				mtf = scenarioc.encode_features(itr_ticker, itr_date, bins, timespan, trial.model_name)

				scenarioc.store_scenarioc_encodings(mtf, itr_ticker, itr_date, timespan, bins)
				_done_count += 1
				print_progress("  Last encoding {} {} [{} Done] [{} Skipped] [{} Weekend/Holiday]".format(itr_ticker, itr_date, _done_count, _skip_count, _err_count))


		## load all label data and feature contexts for batch loading
		_tickers, _dates, _labels = scenarioc.prepare_problemspace(_distinct_tickers, timespan, bins)

		## We'll still want the train data. We'll use it to find the mean and std. deviation of the data.
		_mask_train = (_dates > pd.to_datetime(trial.train_from)) & (_dates < pd.to_datetime(trial.train_until))
		_mask_predict = pd.DataFrame([x for x in zip(_tickers, _dates)])
		_mask_predict = _mask_predict.apply(lambda x: tuple(x) in _contexts, axis=1).values

		_tickers_train = _tickers[_mask_train]
		_dates_train = _dates[_mask_train]
		_tickers_predict = _tickers[_mask_predict]
		_dates_predict = _dates[_mask_predict]
		print("Shapes: [TICKER {}] [DATES {}] [TICKER {}] [DATES {}]".format(_tickers_train.shape[0], _dates_train.shape[0], _tickers_predict.shape[0], _dates_predict.shape[0]))

		print_progress("\n Creating model and loading weights")
		model = scenarioc.create_model(input_shape, filter_shape, output_size, FC_layers)
		model.load_weights("{}/weights{}_{}.h5".format(paths.TEMP_PATH, self.scenario, finetune_path))


		print_progress("\n Finding Mean and Std. Deviation")
		feature_mean, feature_std = scenarioc.features_stats(_dates_train, _tickers_train, _labels, timespan, bins)

		y_preds = scenarioc.predict(model, _dates_predict, _tickers_predict, timespan, bins, feature_mean, feature_std)

		return y_preds




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
			,"modeltrainmarket_status": "INCOMPLETE"
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
		eval_status_df["ticker"] = "PADLINE"
		eval_status_df["epochs"] = 0
		eval_status_df["status"] = "PADLINE"
		eval_status_df["start"] = None
		eval_status_df["end"] = None
		eval_status_df["mse"] = None
		eval_status_df["mse_test"] = None
		eval_status_df["r_squared"] = None
		eval_status_df["accuracy"] = None
		eval_status_df["r_squared_test"] = None
		eval_status_df["accuracy_test"] = None
		eval_status_df["msg"] = None
		eval_status_df.set_index(["ticker", "epochs"], inplace=True)

		encode_status_df = pd.DataFrame()
		encode_status_df["ticker"] = "PADLINE"
		encode_status_df["date"] = None
		encode_status_df["status"] = None
		encode_status_df["start"] = None
		encode_status_df["end"] = None
		encode_status_df["msg"] = None
		encode_status_df.set_index(["ticker", "date"], inplace=True)

		self.trialconfig_df = trialconfig_df
		self.fetchstatus_df = fetchstatus_df
		self.featureengineer_status_df = featureengineer_status_df
		self.train_status_df = train_status_df
		self.eval_status_df = eval_status_df

		self.encode_status_df = encode_status_df
		for i in range(self.encode_workpages):
			self.__dict__["encode_status_df_%d" % i] = encode_status_df.copy()
		
	def store_encodestatus_files(self, work_page):
		self.encode_status_df.to_csv("{}_encode_status_df.tmp".format(self.model_name))

		if(not(work_page is None)):
			self.__dict__["encode_status_df_%d" % work_page].to_csv("{}_encode_status_df_{}.tmp".format(self.model_name, work_page))

	def store_status_files(self, model_name=None):
		model_name = self.model_name if model_name is None else model_name
		self.trialconfig_df.to_csv("{}_trialconfig.tmp".format(model_name))
		self.fetchstatus_df.to_csv("{}_fetchstatus.tmp".format(model_name))
		self.featureengineer_status_df.to_csv("{}_featureengineer_status.tmp".format(model_name))
		self.train_status_df.to_csv("{}_train_status.tmp".format(model_name))
		self.eval_status_df.to_csv("{}_eval_status.tmp".format(model_name))
		self.encode_status_df.to_csv("{}_encode_status_df.tmp".format(model_name))

	def load_status_files(self):
		self.trialconfig_df = pd.read_csv("{}_trialconfig.tmp".format(self.model_name))
		self.fetchstatus_df = pd.read_csv("{}_fetchstatus.tmp".format(self.model_name))
		self.featureengineer_status_df = pd.read_csv("{}_featureengineer_status.tmp".format(self.model_name))
		self.train_status_df = pd.read_csv("{}_train_status.tmp".format(self.model_name))
		self.eval_status_df = pd.read_csv("{}_eval_status.tmp".format(self.model_name))
		self.encode_status_df = pd.read_csv("{}_encode_status_df.tmp".format(self.model_name))
		for i in range(self.encode_workpages):
			self.__dict__["encode_status_df_%d" % i] = pd.read_csv("{}_encode_status_df_{}.tmp".format(self.model_name, i))

		self.trialconfig_df.set_index("property", inplace=True)
		self.fetchstatus_df.set_index("ticker", inplace=True)
		self.featureengineer_status_df.set_index("ticker", inplace=True)
		self.train_status_df.set_index("ticker", inplace=True)
		self.eval_status_df.set_index(["ticker", "epochs"], inplace=True)
		self.encode_status_df.set_index(["ticker", "date"], inplace=True)
		for i in range(self.encode_workpages):
			self.__dict__["encode_status_df_%d" % i].set_index(["ticker", "date"], inplace=True)

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

	def store_scenarioc_features(self, features_df, ticker):
		
		return scenarioc.store_scenarioc_features(features_df, ticker)

	def load_scenarioc_features(self, ticker, parseDate=True):
		
		return scenarioc.load_scenarioc_features(ticker, parseDate)


	def store_scenarioc_encodings(self, feature_data, ticker, date, timespan, bins):
		
		return scenarioc.store_scenarioc_encodings(feature_data, ticker, date, timespan, bins)

	def load_scenarioc_encodings(self, ticker, date, timespan, bins):
		
		return scenarioc.load_scenarioc_encodings(ticker, date, timespan, bins)

	def store_scenarioc_labels(self, features_df, ticker):
		
		return scenarioc.store_scenarioc_labels(features_df, ticker)

	def load_scenarioc_labels(self, ticker, parseDate=True):
		
		return scenarioc.load_scenarioc_labels(ticker, parseDate)


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



