import numpy as np
import pandas as pd
import datetime
from functools import partial
from scipy import stats

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
		temp_df = temp_df.fillna(value=0)

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
		temp_df = temp_df.fillna(value=0)

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
		temp_df = temp_df.fillna(value=0)

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
		temp_df = temp_df.fillna(value=0)

	if merge_result:
		temp_df = pd.concat([orig_df, temp_df], axis=1);

	return temp_df



def calc_beta(df):
	np_array = df.values
	s = np_array[:,0] # stock returns are column zero from numpy array
	m = np_array[:,1] # market returns are column one from numpy array

	slope, intercept, r_value, p_value, std_err = stats.linregress(s, m)

	##covariance = np.cov(s,m) # Calculate covariance between stock and market
	##beta = covariance[0,1]/covariance[1,1]
	return slope

def timewindow_beta(stock_df, market_df, column_slice, period, min_periods=None, set_indexes=True, merge_result=False):
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

		for i in range(1, len(joined_df)+1):
			sub_df = joined_df.iloc[max(i-period, 0):i,:]
			if len(sub_df) >= min_periods:
				idx = sub_df.index[-1]
				itr_beta[idx] = calc_beta(sub_df)

		col_name = "timewindow_beta_{}_{}".format(period, col)
		result[col_name] = itr_beta

	if merge_result:
		if set_indexes:
			result = pd.merge(stock_df, result, left_on='Date', right_index=True)
		else:
			result = pd.concat([stock_df, result], axis=1);

	return result


def calc_alpha(df):
	np_array = df.values
	s = np_array[:,0] # stock returns are column zero from numpy array
	m = np_array[:,1] # market returns are column one from numpy array

	slope, intercept, r_value, p_value, std_err = stats.linregress(s, m)

	##covariance = np.cov(s,m) # Calculate covariance between stock and market
	##beta = covariance[0,1]/covariance[1,1]
	return intercept

def timewindow_alpha(stock_df, market_df, column_slice, period, min_periods=None, set_indexes=True, merge_result=False):
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

		for i in range(1, len(joined_df)+1):
			sub_df = joined_df.iloc[max(i-period, 0):i,:]
			if len(sub_df) >= min_periods:
				idx = sub_df.index[-1]
				itr_beta[idx] = calc_alpha(sub_df)

		col_name = "timewindow_beta_{}_{}".format(period, col)
		result[col_name] = itr_beta

	if merge_result:
		if set_indexes:
			result = pd.merge(stock_df, result, left_on='Date', right_index=True)
		else:
			result = pd.concat([stock_df, result], axis=1);

	return result











