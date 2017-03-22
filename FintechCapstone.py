from __future__ import division, print_function

import math
import uuid
import datetime
import pandas as pd
import numpy as np
from functools import partial
from dateutil import parser as dtparser

from utils import datafetch
from utils import vectorized_funs
from utils import datapipe

CONFIG_PATH = "./config"
DATA_PATH = "./data"
RAW_DATA_PATH = "{}/A_RAW".format(DATA_PATH)
TIER1_DATA_PATH = "{}/B_TIER1".format(DATA_PATH)
TIER2_DATA_PATH = "{}/C_TIER2".format(DATA_PATH)
TINY_FLOAT = 1e-128

class FinCapstone():

	def __init__(self, 
				ticker_list_samplesize=100,
				path_dataload_result="RES_initial_dataload.csv",
				path_ticker_list=None,
				date_from='1900-01-01',
				date_to=str(datetime.date.today()),
				fill_value=1e-128,
				ticker_list=None,
				timespan=None,
				timespan_ab=None):

		self.ticker_list = None
		self.timespan = None
		self.timespan_ab = None
		self.ticker_list_samplesize = ticker_list_samplesize
		self.PATH_DATALOAD_RESULT = path_dataload_result
		self.date_from = date_from
		self.date_to = date_to
		self.fill_value = fill_value

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



	def provision_fulltickerlist(self):
		_rr = None
		_rr = datafetch.load_exchangesinfos()
		_rr = _rr.drop([2922])

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




	def feature_engineering(self):
		## Only work with tickers that were successfull during datafetch
		ticker_list = self.provision_validtickerlist()


		for ix, row in ticker_list.iterrows():
			itr_ticker = row["Symbol"]
			print("\n\n - {} - \n".format(itr_ticker))
			itr_df = datafetch.load_raw_frame(itr_ticker, parseDate=False)

			#Lets get only from 2010 onward
			itr_df = itr_df[pd.to_datetime(itr_df["Date"]) >= dtparser.parse(self.date_from)]
			itr_df = itr_df[pd.to_datetime(itr_df["Date"]) < dtparser.parse(self.date_to)]

			itr_df = self.calc_measures_tier1(itr_df, verbose=True)
			itr_df = self.calc_measures_tier2(itr_df, verbose=True)
			datafetch.store_tier2_frame(itr_df, itr_ticker)




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





	def calc_measures_tier1(self, raw_df, verbose=True):
		"""
			Description:
				<Description>
				
				E.g. : Useful to <...>
			
			Parameters:
				[...]

			Returns : (type)
			
			Examples:
				
		"""

		timespan = None
		timespan = self.timespan

		if verbose:
			self.print_verbose_start()



		## SMAs
		raw_df = vectorized_funs.calc_sma(raw_df, timespan, ["Close", "Volume", "High", "Low", "Open"], merge_result=True);

		if verbose:
			self.print_verbose("SMA")



		## RETURNs
		raw_df = vectorized_funs.calc_return(raw_df, timespan=timespan, column_slice=["Close", "High", "Low", "Volume"], merge_result=True)

		if verbose:
			self.print_verbose("RETURNS")



		## DIFF MOVEs
		raw_df = vectorized_funs.calc_diff_moves(raw_df, timespan=timespan, column_slice=["Close", "High", "Low", "Volume"], merge_result=True)

		if verbose:
			self.print_verbose("DIFF MOVE")



		## BOLLINGER BANDs
		raw_df = vectorized_funs.calc_bollinger(raw_df, timespan, ["Close", "Volume"], merge_result=True, scaler=2)

		if verbose:
			self.print_verbose("BOLLINGER")


		timespan = self.timespan_ab

		## ALPHAs  - THIS COULD BE OPTIMIZED : http://gouthamanbalaraman.com/blog/calculating-stock-beta.html
		## +
		## BETAs  - THIS COULD BE OPTIMIZED : http://stackoverflow.com/a/39503417
		sp500 = datafetch.load_raw_frame("^GSPC")

		for tterm in timespan:
			for t in timespan[tterm]:
				raw_df = vectorized_funs.timewindow_alphabeta(raw_df, sp500, ["Close", "High", "Low"], t, merge_result=True)

		if verbose:
			self.print_verbose("ALPHABETA")
			#self.print_verbose_end()

		return raw_df




	def calc_measures_tier2(self, raw_df, verbose=True):
		"""
			Description:
				<Description>
				
				E.g. : Useful to <...>
			
			Parameters:
				[...]

			Returns : (type)
			
			Examples:
				
		"""

		timespan = None
		timespan = self.timespan


		## AFTERMARKET DIFF
		raw_df = vectorized_funs.calc_aftermarket_diff(raw_df, fillna=True, merge_result=True);

		if verbose:
			self.print_verbose("AM DIFF")



		## AFTERMARKET RETURNs
		raw_df = vectorized_funs.calc_aftermarket_return(raw_df, fillna=True, merge_result=True)

		if verbose:
			self.print_verbose("AM RETURNS")



		## AFTERMARKET RETURNS SMA
		raw_df = vectorized_funs.calc_aftermarket_sma(raw_df, timespan=timespan, merge_result=True)

		if verbose:
			self.print_verbose("AM SMA")



		## AFTERMARKET RETURNS BOLLINGER
		raw_df = vectorized_funs.calc_aftermarket_bollinger(raw_df, timespan, merge_result=True)

		if verbose:
			self.print_verbose("AM BOLLINGER")



		## INTRADAY HIGH-LOW DIFFERENCE
		raw_df = vectorized_funs.calc_highlow_diff(raw_df, merge_result=True)

		if verbose:
			self.print_verbose("HL DIFF")



		## INTRADAY CLOSE-LOW DIFFERENCE
		raw_df = vectorized_funs.calc_closelow_diff(raw_df, merge_result=True)

		if verbose:
			self.print_verbose("CL DIFF")



		## INTRADAY CLOSE-HIGH DIFFERENCE
		raw_df = vectorized_funs.calc_closehigh_diff(raw_df, merge_result=True)

		if verbose:
			self.print_verbose("CH DIFF")



		if verbose:
			self.print_verbose_end()

		return raw_df



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








