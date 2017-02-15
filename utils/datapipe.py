import datetime
import pandas as pd
import numpy as np
from functools import partial
from scipy import stats

from utils.datafetch import *
from utils.vectorized_funs import *

def calc_diff_moves(raw_df, timespan, column_slice, merge_result=True):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	res_df = pd.DataFrame()
	itr_df = pd.DataFrame()

	for ts_name in timespan:
		for t in timespan[ts_name]:
			itr_df = timewindow_diff(raw_df, column_slice=column_slice, shift_window=t, fillna=True, merge_result=False)
			res_df = pd.concat([res_df, itr_df], axis=1)

	if merge_result:
		res_df = pd.concat([raw_df, res_df], axis=1);

	itr_df = None

	return res_df



def calc_return(raw_df, timespan, column_slice, merge_result=True):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	res_df = pd.DataFrame()
	itr_df = pd.DataFrame()

	for ts_name in timespan:
		for t in timespan[ts_name]:
			itr_df = timewindow_return(raw_df, column_slice=column_slice, shift_window=t, merge_result=False, fillna=True)
			res_df = pd.concat([res_df, itr_df], axis=1)

	if merge_result:
		res_df = pd.concat([raw_df, res_df], axis=1);

	return res_df



def calc_sma(raw_df, timespan, column_slice, merge_result=True):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	res_df = pd.DataFrame()
	itr_df = pd.DataFrame()

	for ts_name in timespan:
		for t in timespan[ts_name]:
			itr_df = roll_columns(raw_df, "mean", column_slice=column_slice, window=t, merge_result=False, pad_result=True)
			res_df = pd.concat([res_df, itr_df], axis=1)

	if merge_result:
		res_df = pd.concat([raw_df, res_df], axis=1);

	return res_df


def calc_bollinger(raw_df, timespan, column_slice, merge_result=True, scaler=2):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	itr_df = pd.DataFrame()
	res_df = pd.DataFrame()

	for col in column_slice:
		for ts_name in timespan:
			if ts_name in ["medium_term", "long_term"]:
				for t in timespan[ts_name]:
					itr_df = roll_columns(raw_df, "std", column_slice=[col], window=t, merge_result=False, scaler=scaler, pad_result=True)
					upper_band = itr_df.apply(lambda x: raw_df[col] + x)
					lower_band = itr_df.apply(lambda x: raw_df[col] - x)
					
					roll_name = "{}_roll_2std_{}".format("Close", t)
					upper_name = "{}_bollinger_{}_up".format("Close", t)
					lower_name = "{}_bollinger_{}_low".format("Close", t)
					
					res_df[roll_name] = itr_df.iloc[:, 0]
					res_df[upper_name] = upper_band
					res_df[lower_name] = lower_band

	if merge_result:
		res_df = pd.concat([raw_df, res_df], axis=1);

	return res_df


def calc_measures_tier1(raw_df, verbose=True):
	timespan = dict();

	# timespan = {
	# 	"short_term": [1, 2, 5]
	#     ,"medium_term": [10, 30, 40, 70, 90]
	#     ,"long_term": [100, 200, 300, 400]
	# }

	_start = None
	_end = None

	_step_i = None
	_step_f = None

	timespan = {
		"short_term": [1,3,5]
	    ,"medium_term": [40, 60]
	    ,"long_term": [90, 150, 220]
	}

	_start = datetime.datetime.now()
	_step_i = _start

	if verbose:
		print("| | START     - {}".format(str(_start)))



	## SMAs
	raw_df = calc_sma(raw_df, timespan, ["Close", "Volume", "High", "Low", "Open"], merge_result=True);

	if verbose:
		_step_f = datetime.datetime.now()
		print("\ / SMA       - {}".format(str(_step_f - _step_i)))
		_step_i = _step_f



	## RETURNs
	raw_df = calc_return(raw_df, timespan=timespan, column_slice=["Close", "High", "Low", "Volume"], merge_result=True)

	if verbose:
		_step_f = datetime.datetime.now()
		print("\ / RETURNS   - {}".format(str(_step_f - _step_i)))
		_step_i = _step_f



	## DIFF MOVEs
	raw_df = calc_diff_moves(raw_df, timespan=timespan, column_slice=["Close", "High", "Low", "Volume"], merge_result=True)

	if verbose:
		_step_f = datetime.datetime.now()
		print("\ / DIFF MOVE - {}".format(str(_step_f - _step_i)))
		_step_i = _step_f



	## BOLLINGER BANDs
	raw_df = calc_bollinger(raw_df, timespan, ["Close", "Volume"], merge_result=True, scaler=2)

	if verbose:
		_step_f = datetime.datetime.now()
		print("\ / BOLLINGER - {}".format(str(_step_f - _step_i)))
		_step_i = _step_f


	timespan = {
		"short_term": [5]
	    ,"medium_term": [30]
	    ,"long_term": []
	}

	## ALPHAs  - THIS COULD BE OPTIMIZED : http://gouthamanbalaraman.com/blog/calculating-stock-beta.html
	## +
	## BETAs  - THIS COULD BE OPTIMIZED : http://stackoverflow.com/a/39503417
	sp500 = load_raw_frame("^GSPC")

	for tterm in timespan:
		for t in timespan[tterm]:
			raw_df = timewindow_beta(raw_df, sp500, ["Close", "High", "Low"], t, merge_result=True)

	if verbose:
		_step_f = datetime.datetime.now()
		print("\ / ALPHABETA - {}".format(str(_step_f - _step_i)))
		_end = _step_f
		print(" V  END       - {} (TOOK {})".format(str(_end), str(_end - _start)))

	return raw_df




