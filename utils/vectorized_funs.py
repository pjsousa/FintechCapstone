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
			
	"""

	itr_df = pd.DataFrame()
	res_df = pd.DataFrame()

	for col in column_slice:
		for ts_name in timespan:
			if ts_name in ["medium_term", "long_term"]:
				for t in timespan[ts_name]:
					itr_movingavg = roll_columns(raw_df, "mean", column_slice=[col], window=t, merge_result=False, pad_result=True)

					itr_df = roll_columns(raw_df, "std", column_slice=[col], window=t, merge_result=False, scaler=scaler, pad_result=True)
					upper_band = itr_df.apply(lambda x: itr_movingavg["{}_roll_mean_{}".format(col,t)] + x)
					lower_band = itr_df.apply(lambda x: itr_movingavg["{}_roll_mean_{}".format(col,t)] - x)

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



def digitize_multitonic(values, bins):
	"""
		Done like this.
		https://docs.google.com/spreadsheets/d/1O6mcYoKiiMCbyM3V_tztd94XPGyw6rU2Fghhokq6kVs/edit#gid=1323898421
	"""

	print(values.name)
	print(values.shape)
	values = values.values.reshape((values.shape[0], 1))

	print(values.shape)
	print(bins.shape)

	diffs = np.subtract(values, bins)
	diffs = pd.DataFrame(diffs)

	signs = np.sign(diffs)

	def findfirst(row, value=0):
		_r = row[row == value]
		if _r.shape[0] == 0:
			_r = -1
		else:
			_r = _r.index[0]
		return _r

	pos = pd.DataFrame(columns=["<0", "=0",">0"])
	pos["<0"] = signs.apply(partial(findfirst, value=-1), axis=1)
	pos["=0"] = signs.apply(partial(findfirst, value=0), axis=1)
	pos[">0"] = signs.apply(partial(findfirst, value=1), axis=1)
	pos.loc[(pos["<0"] == -1) & (pos["=0"] == -1), [">0"]] = -1

	has = pd.DataFrame(columns=["<0", "=0",">0"])
	has["<0"] = (signs == -1).astype(np.int).apply(np.sum, axis=1)
	has["=0"] = (signs == 0).astype(np.int).apply(np.sum, axis=1)
	has[">0"] = (signs == 1).astype(np.int).apply(np.sum, axis=1)
	has = np.clip(has, 0, 1)
	

	monotonic_temp = pd.DataFrame(bins[pos], columns=["<0","=0",">0"])
	def monotonic_r(row):
		_r = None
		
		if row["=0"] == 1:
			_r = [0, 1, 0]
		elif row["<0"] == 1:
			_r = [1, 0, 0]
		else:
			_r = [0, 0, 1]
		return pd.Series(_r)

	mono_r = np.sum(monotonic_temp * has.apply(monotonic_r, axis=1).values, axis=1)




	monotonic_temp2 = pd.DataFrame(pos, columns=["<0","=0",">0"])
	monotonic_temp2["<0"] = pos["<0"] - (pos > 0).astype(np.int)["<0"]
	monotonic_temp2 = pd.DataFrame(bins[monotonic_temp2], columns=["<0","=0",">0"])
	
	monotonic_temp2.loc[np.array([17,18,19],), :]

	def monotonic_l(row):
		_r = None
		
		if row["=0"] == 1:
			_r = [0, 1, 0]
		elif row["<0"] == 1:
			_r = [1, 0, 0]
		else:
			_r = [0, 0, 1]
		return pd.Series(_r)

	mono_l = np.sum(monotonic_temp2 * has.apply(monotonic_l, axis=1).values, axis=1)


	values_sign = np.sign(values)

	def bitonic_0(row):
		_r = None
		
		if row["=0"] == 1:
			_r = mono_r[row_name]
		else:
			if values_sign[row.name] == -1:
				_r = mono_r[row.name]
			else:
				_r = mono_l[row.name]
		return _r

	#bit_0 = np.sum(monotonic_temp2 * has.apply(monotonic_r, axis=1).values, axis=1)

	def bitonic_infinite(row):
		_r = None
		
		if row["=0"] == 1:
			_r = mono_r[row_name]
		else:
			if values_sign[row.name] == -1:
				_r = mono_l[row.name]
			else:
				_r = mono_r[row.name]
		return _r

	bit_inf = np.sum(monotonic_temp2 * has.apply(monotonic_r, axis=1).values, axis=1)

	return bit_inf
	#return pd.DataFrame(np.array([mono_r, mono_l, bit_0, bit_inf]).T, columns=["mono_r", "mono_l", "bit_0", "bit_inf"], index=values)



def calc_discretize(raw_df, bins, column_slice=None, merge_result=True):
	_r = None

	if column_slice is None:
		column_slice = raw_df.columns.tolist()

	_r = []

	for col in column_slice:
		inds_gain = digitize_multitonic(raw_df[col], bins)
		_r.append(inds_gain.values)

	_r = pd.DataFrame(np.array(_r).T, columns=["discrete_{}".format(x) for x in column_slice])

	if merge_result:
		_r = pd.concat([raw_df, _r], axis=1);

	return _r



def rsiFunc(prices, n=14):
	deltas = np.diff(prices)
	seed = deltas[:n+1]
	up = seed[seed>=0].sum()/n
	down = -seed[seed<0].sum()/n
	rs = up/down
	rsi = np.zeros_like(prices)
	rsi[:n] = 100.0 - 100./(1. + rs)
	
	for i in range(n, len(prices)):
		delta = deltas[i-1]
		if delta > 0:
			upval = delta
			downval = 0.
		else:
			upval = 0.
			downval = -delta
		
		up = (up*(n-1) + upval) / n
		down = (down*(n-1)+downval)/n
		
		rs = up/down
		rsi[i] = 100. - 100 / (1.+rs)
	
	return rsi



def onbalancevolumeFunc(daily_returns, volume):
	_r = None
	
	_signal = daily_returns.copy()
	_signal.where(_signal >= 0, -1, inplace=True)
	_signal.where(_signal < 0, 1, inplace=True)
	
	_r = volume * _signal

	return _r.cumsum()



def ExpMovingAverage(values, window):
	weights = np.exp(np.linspace(-1., 0., window))
	weights = weights / weights.sum()
	a = np.convolve(values, weights, mode="full")[:values.shape[0]]
	a[:window] = a[window]
	return a



def calc_macd(x, slow=26, fast=12):
	emaslow = ExpMovingAverage(x, slow)
	emafast = ExpMovingAverage(x, fast)
	return emaslow, emafast, emafast - emaslow



def calc_stochasticoscilator(raw_df, window=14):
	max_hi = raw_df["High"].rolling(window=window).max()
	min_lo = raw_df["Low"].rolling(window=window).min()

	return (raw_df["Close"] - min_lo) / (max_hi - min_lo) * 100.0



def calc_adx(raw_df):
	yesterdayclose = raw_df["Close"].shift(1)

	atr_df = pd.DataFrame()
	atr_df["H-L"] = raw_df["High"] - raw_df["Low"]
	atr_df["H-YC"] = np.abs(raw_df["High"] - yesterdayclose)
	atr_df["L-YC"] = np.abs(raw_df["Low"] - yesterdayclose)
	atr_df["TR"] = atr_df.max(axis=1)
	atr = ExpMovingAverage(atr_df["TR"], 14)

	moveup = raw_df["High"] - raw_df["High"].shift(1)
	movedown = raw_df["Low"].shift(1) - raw_df["Low"]


	pdm = moveup.where((moveup <= 0) | (moveup <= movedown), 0)
	ndm = movedown.where((movedown <= 0) | (movedown <= moveup), 0)

	pdm_ema14 = ExpMovingAverage(pdm, 14)
	ndm_ema14 = ExpMovingAverage(ndm, 14)

	pdi = (pdm_ema14 / atr) * 100.0
	ndi = (ndm_ema14 / atr) * 100.0

	adx = 100.0 * ExpMovingAverage(np.abs((pdi - ndi)), 14) / (pdi + ndi)
	
	return adx, pdi, ndi



def time_sincelastmax(windowframe, aroon_df):
	argmax_h = np.argmax(aroon_df["H"].iloc[windowframe.astype(np.int)])
	#argmax_date = aroon_df.loc[argmax_h, "Date"]
	#today_date = aroon_df.loc[windowframe[-1], "Date"]
	return (windowframe[-1] - argmax_h)



def time_sincelastmin(windowframe, aroon_df):
	argmin_h = np.argmin(aroon_df["L"].iloc[windowframe.astype(np.int)])
	#argmin_date = aroon_df.loc[argmin_h, "Date"]
	#today_date = aroon_df.loc[windowframe[-1], "Date"]
	return (windowframe[-1] - argmin_h)



def calc_aroon(raw_df, window=25):

	aroon_df = pd.DataFrame()
	aroon_df["Date"]= pd.Series(raw_df.index.tolist())
	aroon_df["H"] = raw_df["High"].values
	aroon_df["LAST_H"] = pd.Series(np.arange(aroon_df.shape[0]).astype(np.int32)).rolling(window=window).apply(time_sincelastmax, args=[aroon_df])
	aroon_df["L"] = raw_df["Low"].values
	aroon_df["LAST_L"] = pd.Series(np.arange(aroon_df.shape[0]).astype(np.int32)).rolling(window=window).apply(time_sincelastmin, args=[aroon_df])
	aroon_df["AROON_UP"] = ((window - aroon_df["LAST_H"]) / float(window)) * 100.0
	aroon_df["AROON_DOWN"] = ((window - aroon_df["LAST_L"]) / float(window)) * 100.0
	
	return aroon_df["AROON_UP"], aroon_df["AROON_DOWN"]



def calc_chaikin_money_flow(raw_df, window=21):
	dmf = (((raw_df["Close"] - raw_df["Low"]) - (raw_df["High"] - raw_df["Close"])) / (raw_df["High"] - raw_df["Low"])) * raw_df["Volume"]
	cmf = dmf.rolling(window=window).mean() / raw_df["Volume"].rolling(window=window).mean()
	return cmf, dmf
