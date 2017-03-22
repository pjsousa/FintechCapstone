import numpy as np
import pandas as pd
import datetime
from functools import partial
from scipy import stats

TINY_FLOAT = 1e-128

def roll_columns(orig_df, stat_fun_name, column_slice=None, window=None, merge_result=False, pad_result=False, scaler=1):
	"""
		Description:
			Calculates a rolling statistic over one or multiple columns of a dataframe.
			
			E.g. : Useful to calculate the SMA 10 for Close;
				   Useful to calculate the SMA 30 for Close and Volume;
		
		Parameters:
			orig_df : (DataFrame) the original dataframe to work on
			stat_fun_name : (string) the name of the statistic to roll
			column_slice : (array) an array of column names to roll 
				- defaults to all the columns on the dataframe
			window : (int) the row window to roll over
				- defaults to 10
				- positive values will consider the last 'window' rows
				- negative values will consider the next 'window' rows
			merge_result : (bool) whether or not to merge the resulting Dataframe into the 'orig_df'
				- when True, merges and returns the full dataframe
				- when False, does not merge and returns only the calculated dataframe
			pad_result : (bool) whether or not to pad the rolling limits
			scaler : (Number) a scaler value to multiply the calculated DataFrame
				- defauls to 1

		Returns : (DataFrame)
			When merge_result=False, returns a DataFrame with the calculated statistics
			When merge_result=True, merges the calculated statistics to orig_df and returns that
		
		Examples:
			
	"""

	if column_slice is None:
		column_slice = orig_df.columns.tolist()

	if window is None:
		window = 10

	temp_df = None
	fn_rolling = None

	fn_rolling = orig_df.loc[ : , column_slice].rolling(window=window, center=False)
	temp_df = getattr(fn_rolling, stat_fun_name)()
	temp_df.columns = ["{}_roll_{}_{}".format(x, stat_fun_name, window) for x in column_slice]

	if pad_result:
		# This one is happening inplace
		temp_df.fillna(method="bfill", axis=0, inplace=True)

	if scaler != 1:
		temp_df = temp_df * scaler

	if merge_result:
		temp_df = pd.concat([orig_df, temp_df], axis=1);

	return temp_df


def timewindow_diff(orig_df, column_slice=None, shift_window=1, fillna=False, merge_result=False):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""
	if column_slice is None:
		column_slice = orig_df.columns.tolist()

	temp_df = orig_df.loc[ : , column_slice] - orig_df.loc[ : , column_slice].shift(shift_window)
	temp_df.columns = ["timewindow_diff_{}_{}".format(shift_window,x) for x in column_slice]

	if fillna:
		temp_df = temp_df.fillna(value=TINY_FLOAT)

	if merge_result:
		temp_df = pd.concat([orig_df, temp_df], axis=1);

	return temp_df


def timewindow_return(orig_df, column_slice=None, shift_window=1, fillna=False, merge_result=False):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	if column_slice is None:
		column_slice = orig_df.columns.tolist()

	temp_df = (orig_df.loc[ : , column_slice] / orig_df.loc[ : , column_slice].shift(shift_window)) - 1
	temp_df.columns = ["timewindow_return_{}_{}".format(shift_window,x) for x in column_slice]

	if fillna:
		temp_df = temp_df.fillna(value=TINY_FLOAT)

	if merge_result:
		temp_df = pd.concat([orig_df, temp_df], axis=1);

	return temp_df


def timewindow_cumdiff(orig_df, column_slice=None, fillna=False, merge_result=False):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	if column_slice is None:
		column_slice = orig_df.columns.tolist()

	temp_df = orig_df.loc[ : , column_slice] - orig_df.loc[ 0 , column_slice]
	temp_df.columns = ["timewindow_cumdiff_{}".format(x) for x in column_slice]

	if fillna:
		temp_df = temp_df.fillna(value=TINY_FLOAT)

	if merge_result:
		temp_df = pd.concat([orig_df, temp_df], axis=1);

	return temp_df


def timewindow_cumreturn(orig_df, column_slice=None, fillna=False, merge_result=False):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""
	
	if column_slice is None:
		column_slice = orig_df.columns.tolist()

	temp_df = (orig_df.loc[ : , column_slice] / orig_df.loc[ 0 , column_slice]) - 1
	temp_df.columns = ["timewindow_cumreturn_{}".format(x) for x in column_slice]

	if fillna:
		temp_df = temp_df.fillna(value=TINY_FLOAT)

	if merge_result:
		temp_df = pd.concat([orig_df, temp_df], axis=1);

	return temp_df


def calc_alphabeta(df):
	np_array = df.values
	s = np_array[:,0] # stock returns are column zero from numpy array
	m = np_array[:,1] # market returns are column one from numpy array

	slope, intercept, r_value, p_value, std_err = stats.linregress(s, m)

	##covariance = np.cov(s,m) # Calculate covariance between stock and market
	##beta = covariance[0,1]/covariance[1,1]
	return slope, intercept


def timewindow_alphabeta(stock_df, market_df, column_slice, period, min_periods=None, set_indexes=True, merge_result=False, return_alpha=True, return_beta=True, fillna=True):
	if min_periods is None:
		min_periods = period
	
	if set_indexes:
		stock_i = stock_df.set_index("Date")
		market_i = market_df.set_index("Date")
	else:
		stock_i = stock_df
		market_i = market_df

	result = pd.DataFrame(index=stock_i.index)

	for col in column_slice:
		joined_df = pd.merge(stock_i.loc[:,[col]], market_i.loc[:,[col]], how="left", left_index=True, right_index=True)
		
		itr_beta = pd.Series(np.nan, index=joined_df.index)
		itr_alpha = pd.Series(np.nan, index=joined_df.index)

		for i in range(1, len(joined_df)+1):
			sub_df = joined_df.iloc[max(i-period, 0):i,:]
			if len(sub_df) >= min_periods:
				idx = sub_df.index[-1]
				slope, intercept = calc_alphabeta(sub_df)
				itr_beta[idx] = slope
				itr_alpha[idx] = intercept

		if return_beta:
			col_name = "timewindow_beta_{}_{}".format(period, col)
			result[col_name] = itr_beta
		
		if return_alpha:
			col_name = "timewindow_alpha_{}_{}".format(period, col)
			result[col_name] = itr_alpha


	if merge_result:
		if set_indexes:
			result = pd.merge(stock_df, result, left_on='Date', right_index=True)
		else:
			result = pd.concat([stock_df, result], axis=1);

	if fillna:
		result.fillna(TINY_FLOAT, inplace=True)

	return result


def calc_diff_moves(raw_df, timespan, column_slice, merge_result=True, fillna=True):
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
					
					roll_name = "{}_roll_2std_{}".format(col, t)
					upper_name = "{}_bollinger_{}_up".format(col, t)
					lower_name = "{}_bollinger_{}_low".format(col, t)
					
					res_df[roll_name] = itr_df.iloc[:, 0]
					res_df[upper_name] = upper_band
					res_df[lower_name] = lower_band

	if merge_result:
		res_df = pd.concat([raw_df, res_df], axis=1);

	return res_df



def calc_aftermarket_diff(raw_df, fillna=True, merge_result=False):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	_r = None
	temp_ser = (raw_df["Open"] - raw_df["Close"].shift(1))
	
	if fillna:
		temp_ser = temp_ser.fillna(value=TINY_FLOAT)

	if merge_result:
		raw_df["aftermarket_diff"] = temp_ser
		_r = raw_df
	else:
		_r = pd.DataFrame(temp_ser, columns=["aftermarket_diff"])

	return _r



def calc_aftermarket_return(raw_df, fillna=True, merge_result=False):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	_r = None
	temp_ser = ((raw_df["Open"] / raw_df["Close"].shift(1))) - 1
	
	if fillna:
		temp_ser = temp_ser.fillna(value=TINY_FLOAT)

	if merge_result:
		raw_df["aftermarket_return"] = temp_ser
		_r = raw_df
	else:
		_r = pd.DataFrame(temp_ser, columns=["aftermarket_return"])

	return _r



def calc_aftermarket_sma(raw_df, timespan, merge_result=False):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	afm_ret = None
	res_df = None
	
	if not ( "aftermarket_return" in raw_df.columns) :
		afm_ret = calc_aftermarket_return(raw_df, merge_result=False)
	else:
		afm_ret = raw_df

	res_df = calc_sma(afm_ret, timespan, ["aftermarket_return"], merge_result=False);

	if merge_result:
		res_df = pd.concat([raw_df, res_df], axis=1);

	return res_df



def calc_aftermarket_bollinger(raw_df, timespan, merge_result=False):
	"""
		Description:
			<Description>
			
			E.g. : Useful to <...>
		
		Parameters:
			[...]

		Returns : (type)
		
		Examples:
			
	"""

	afm_ret = None
	res_df = None
	
	if not ( "aftermarket_return" in raw_df.columns) :
		afm_ret = calc_aftermarket_return(raw_df, merge_result=False)
	else:
		afm_ret = raw_df

	res_df = calc_bollinger(afm_ret, timespan, ["aftermarket_return"], merge_result=False, scaler=2)

	if merge_result:
		res_df = pd.concat([raw_df, res_df], axis=1);

	return res_df



def calc_highlow_diff(raw_df, merge_result=False):
	_r = None
	temp_ser = raw_df["High"] - raw_df["Low"]

	if merge_result:
		raw_df["hilo_diff"] = temp_ser
		_r = raw_df
	else:
		_r = pd.DataFrame(temp_ser, columns=["hilo_diff"])

	return _r



def calc_closelow_diff(raw_df, merge_result):
	_r = None
	temp_ser = raw_df["Close"] - raw_df["Low"]

	if merge_result:
		raw_df["closelo_diff"] = temp_ser
		_r = raw_df
	else:
		_r = pd.DataFrame(temp_ser, columns=["closelo_diff"])

	return _r


def calc_closehigh_diff(raw_df, merge_result):
	_r = None
	temp_ser = raw_df["High"] - raw_df["Close"]

	if merge_result:
		raw_df["closehi_diff"] = temp_ser
		_r = raw_df
	else:
		_r = pd.DataFrame(temp_ser, columns=["closehi_diff"])

	return _r
