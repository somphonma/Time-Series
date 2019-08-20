import sklearn.model_selection
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.interpolate import CubicSpline
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import datetime
from random import randrange
from math import sqrt
import statsmodels
import datetime as dt
from scipy import interpolate
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import datetime
import re
from datetime import datetime
from datetime import timedelta
import os


class Forecasting:

    time = {'day': 'D', 'week': 'W', 'month': 'MS', 'quarter': 'QS', 'year': 'AS'}
    frequency = {'day': 365, 'week': 52, 'month': 12, 'quarter': 4, 'year': 1}

    normal_time = ['year', 'quarter', 'month', 'week', 'day']

    def __init__(self, filename, column_no, forecast_month):
        self.filename = filename
        self.column_no = column_no
        self.no_of_forecast_month = forecast_month
        self.data = pd.read_csv(filename, index_col=0)
        self.data.index = pd.to_datetime(self.data.index)
        self.name = self.data.columns[self.column_no - 1]
    @staticmethod
    def x13(s):
        result=sm.tsa.x13_arima_analysis(s, freq=12, outlier=None)
        return result

    def percent_change(self, data, typ_time):

        change_percent = data.pct_change(periods=self.frequency[typ_time])
        return change_percent

    @staticmethod
    def volumn_change(x, x_vol):

        x = x.dropna(how='any')
        x, x_vol = np.array(x),np.array(x_vol)
        for i in range(12,len(x_vol)):
            t = np.array([x[i-12]+x[i-12]*(x_vol[i]/100)])
            x = np.append(x, t)
        print(x)

    @staticmethod
    def seasonal_decomposes(s):

        result = seasonal_decompose(s, model='additive', freq=12)
        return result

    @staticmethod
    def outlier(history, upper=1, lower=0):

        lower = history.quantile(lower)
        upper = history.quantile(upper)
        k = history.mask(history >= upper)
        history = k.mask(k <= lower)
        return history

    def group_by(df, myList=[], *args):
        return df.groupby(myList)

    def resemble_week(self, data):
        value_in_week = list(data.groupby(data['week_order'])['A'].sum())
        first_day = datetime.date(data.index[0].year, data.index[0].month, data.index[0].day)
        week_date = pd.to_datetime(first_day)
        week_date_list = []
        while week_date < data.index[len(data) - 1]:
            week_date_list.append(week_date)
            week_date = week_date + relativedelta(days=7)
            if week_date.day > 22:
                if week_date.month == 12:
                    week_date = datetime.date(week_date.year + 1, 1, 1)
                    week_date = pd.to_datetime(week_date)
                else:
                    week_date = datetime.date(week_date.year, week_date.month + 1, 1)
                    week_date = pd.to_datetime(week_date)
        future_forecast = pd.DataFrame(value_in_week, index=week_date_list, columns=['BMIX_E'])
        return future_forecast

    # for quarter and week
    def percentage_type2(self, data, input_period, train_period):

        lst = self.time_group(input_period, train_period)
        lst.append(train_period)
        converted_data = data.groupby(lst)[self.name].sum().reset_index()
        converted_data['total'] = converted_data.groupby([converted_data[lst[len(lst)-1]]])[self.name].transform(sum)
        #percentage = converted_data[self.name] / converted_data['total']
        converted_data[self.name] = converted_data[self.name] / converted_data['total']
        return converted_data
    # for day month and year

    def percentage_type1(self, data, input_period, train_period):
        if input_period == train_period:
            lst=[input_period]
        else:
            lst = self.time_group(input_period, train_period)
        converted_data = data.groupby(lst)[self.data.columns[self.column_no-1]].sum()
        percentage = converted_data / converted_data.sum()
        return percentage

    @staticmethod
    def data_info(data):
        data['day'] = data.index.day
        data['month'] = data.index.month
        data['year'] = data.index.year
        data['quarter'] = data.index.quarter
        data['week'] = data.index.day // 7.2 + 1
        data['week'] = data['week'].astype(int)
        maxVal = 4
        data['week'] = data['week'].where(data['week'] <= maxVal, maxVal)
        data['week_order'] = (data['year']-data.index[0].year)*48+(data['month'] - 1) * 4 + data['week']
        data['quarter_order'] = (data['week_order'] // 13) + 1
        return data

    def extract_time_type(self, date):

        time_type = []
        for i, j in self.time.items():
            if i == date:
                time_type.extend([i, j])
                return time_type
        print('the input must be type of datetime for example day,week,month,quarter,year')

    def time_group(self, input_time, output_time):
        lst = self.normal_time[self.normal_time.index(output_time)+1:self.normal_time.index(input_time)+1]
        print(lst)
        return lst

    # single
    def expos(self, hi, lo,input_period, train_period, output_period, pct_require):

        input_time = self.extract_time_type(input_period)
        train_time = self.extract_time_type(train_period)

        # extract column from data using column_no
        data = self.data.iloc[:, self.column_no - 1].dropna()
        data = pd.DataFrame(data)
        # convert from month to year
        if self.frequency[input_time[0]] >= self.frequency[train_time[0]]:
            data_check = data.copy()
            if train_period == 'week':
                previous_period = 'week'
            if train_period == 'month':
                previous_period = 'day'
            if train_period == 'quarter':
                previous_period = 'quarter'
            if train_period == 'year':
                previous_period = 'month'
            last_time = getattr(data.index[len(data) - 1], previous_period)
            # cutting data if it's less than 6
            if last_time > self.frequency[input_period] / 2:
                data = self.arima_to_fill(data, last_time, self.frequency[previous_period], input_period)
            else:
                data = self.remove(data, last_time)
            data_info = self.data_info(data)
            # get portion
            if train_period == 'month' or train_period == 'year':
                percent_portion = self.percentage_type1(data_info, input_time[0], train_time[0]).reset_index()
            else:
                percent_portion = self.percentage_type2(data_info, input_time[0], train_time[0]).reset_index()
        else:
            print(input_time, ' cannot be converted to ', train_time, ' as it is smaller')

        # if(train_period=='week'):
        if train_period == 'week':
            data = self.resemble_week(data_info[self.name])
        data = data[self.name].resample(self.time[train_period]).sum()
        if pct_require:
            data = self.percent_change(data, train_period).dropna()
        train = self.outlier(data, lower=lo, upper=hi)
        train = train.interpolate(method='cubic', order=3, s=0).ffill().bfill()
        model = SimpleExpSmoothing(train)
        model_fit = model.fit()
        future_forecast = model_fit.predict(len(train), len(train) + self.no_of_forecast_month - 1).reset_index()
        future_forecast = future_forecast.rename(columns={'index': 'Date', 0:self.name})
        train = train.reset_index()
        train = train.append(future_forecast)
        # train = self.percent_change(train).reset_index()
        # train = pd.melt(train, id_vars=[train.columns[0]], value_vars=[train.columns[1]])

        train = train[len(train) - self.no_of_forecast_month:len(train)].reset_index(drop=True)
        time_add = train_period + 's'
        if time_add == 'quarters':
            last_date = train['Date'][len(train) - 1] + relativedelta(months=4)
        else:
            last_date = train['Date'][len(train) - 1] + relativedelta(**{time_add: 1})
        dummy_data = pd.DataFrame(columns=['Date', self.name])
        dummy_data.loc[0] = [last_date, 0]
        train = train.append(dummy_data)
        train = train.set_index('Date')
        train_converted = train[self.name].resample(self.time[input_period], fill_method='ffill')
        train_converted = train_converted[:-1].reset_index()
        portion = percent_portion
        if self.frequency[input_time[0]] == self.frequency[train_time[0]]:
            portion[self.name] = 1
        train_converted['Date'] = pd.to_datetime(train_converted['Date'])
        train_converted['month'] = pd.DatetimeIndex(train_converted['Date']).month
        join_data = train_converted.merge(percent_portion, left_on=input_period, right_on=input_period)
        join_data[self.name] = join_data[self.name + '_x'] * join_data[self.name + '_y']
        join_data = join_data.sort_values(by=['Date']).set_index('Date')
        forecast_value = pd.Series(join_data[self.name], index=join_data.index)
        final = forecast_value.resample(self.time[output_period]).sum().reset_index()
        final = pd.melt(final, id_vars=[final.columns[0]], value_vars=[final.columns[1]])
        print(final)

    def double_expos(self, hi, lo,input_period, train_period, output_period, pct_require):

        input_time = self.extract_time_type(input_period)
        train_time = self.extract_time_type(train_period)

        # extract column from data using column_no
        data = self.data.iloc[:, self.column_no - 1].dropna()
        data = pd.DataFrame(data)
        # convert from month to year
        if self.frequency[input_time[0]] >= self.frequency[train_time[0]]:
            data_check = data.copy()
            if train_period == 'week':
                previous_period = 'week'
            if train_period == 'month':
                previous_period = 'day'
            if train_period == 'quarter':
                previous_period = 'quarter'
            if train_period == 'year':
                previous_period = 'month'
            last_time = getattr(data.index[len(data) - 1], previous_period)
            # cutting data if it's less than 6
            if last_time > self.frequency[input_period] / 2:
                data = self.arima_to_fill(data, last_time, self.frequency[previous_period], input_period)
            else:
                data = self.remove(data, last_time)
            data_info = self.data_info(data)
            # get portion
            if train_period == 'month' or train_period == 'year':
                percent_portion = self.percentage_type1(data_info, input_time[0], train_time[0]).reset_index()
            else:
                percent_portion = self.percentage_type2(data_info, input_time[0], train_time[0]).reset_index()
        else:
            print(input_time, ' cannot be converted to ', train_time, ' as it is smaller')

        # if(train_period=='week'):
        if train_period == 'week':
            data = self.resemble_week(data_info[self.name])
        data = data[self.name].resample(self.time[train_period]).sum()
        if pct_require:
            data = self.percent_change(data, train_period).dropna()
        train = self.outlier(data, lower=lo, upper=hi)
        train = train.interpolate(method='cubic', order=3, s=0).ffill().bfill()
        model = ExponentialSmoothing(train, trend='add', seasonal=None)
        model_fit = model.fit()
        future_forecast = model_fit.predict(len(train), len(train) + self.no_of_forecast_month - 1).reset_index()
        future_forecast = future_forecast.rename(columns={'index': 'Date', 0:self.name})
        train = train.reset_index()
        train = train.append(future_forecast)
        # train = self.percent_change(train).reset_index()
        # train = pd.melt(train, id_vars=[train.columns[0]], value_vars=[train.columns[1]])

        train = train[len(train) - self.no_of_forecast_month:len(train)].reset_index(drop=True)
        time_add = train_period + 's'
        if time_add == 'quarters':
            last_date = train['Date'][len(train) - 1] + relativedelta(months=4)
        else:
            last_date = train['Date'][len(train) - 1] + relativedelta(**{time_add: 1})
        dummy_data = pd.DataFrame(columns=['Date', self.name])
        dummy_data.loc[0] = [last_date, 0]
        train = train.append(dummy_data)
        train = train.set_index('Date')
        train_converted = train[self.name].resample(self.time[input_period], fill_method='ffill')
        train_converted = train_converted[:-1].reset_index()
        portion = percent_portion
        if self.frequency[input_time[0]] == self.frequency[train_time[0]]:
            portion[self.name] = 1
        train_converted['Date'] = pd.to_datetime(train_converted['Date'])
        train_converted['month'] = pd.DatetimeIndex(train_converted['Date']).month
        join_data = train_converted.merge(percent_portion, left_on=input_period, right_on=input_period)
        join_data[self.name] = join_data[self.name + '_x'] * join_data[self.name + '_y']
        join_data = join_data.sort_values(by=['Date']).set_index('Date')
        forecast_value = pd.Series(join_data[self.name], index=join_data.index)
        final = forecast_value.resample(self.time[output_period]).sum().reset_index()
        final = pd.melt(final, id_vars=[final.columns[0]], value_vars=[final.columns[1]])
        print(final)
    # additive holtwinter

    def hw(self, hi, lo,input_period, train_period, output_period, pct_require):

        input_time = self.extract_time_type(input_period)
        train_time = self.extract_time_type(train_period)

        # extract column from data using column_no
        data = self.data.iloc[:, self.column_no - 1].dropna()
        data = pd.DataFrame(data)
        # convert from month to year
        if self.frequency[input_time[0]] >= self.frequency[train_time[0]]:
            data_check = data.copy()
            if train_period == 'week':
                previous_period = 'week'
            if train_period == 'month':
                previous_period = 'day'
            if train_period == 'quarter':
                previous_period = 'quarter'
            if train_period == 'year':
                previous_period = 'month'
            last_time = getattr(data.index[len(data) - 1], previous_period)
            # cutting data if it's less than 6
            if last_time > self.frequency[input_period] / 2:
                data = self.arima_to_fill(data, last_time, self.frequency[previous_period], input_period)
            else:
                data = self.remove(data, last_time)
            data_info = self.data_info(data)
            # get portion
            if train_period == 'month' or train_period == 'year':
                percent_portion = self.percentage_type1(data_info, input_time[0], train_time[0]).reset_index()
            else:
                percent_portion = self.percentage_type2(data_info, input_time[0], train_time[0]).reset_index()
        else:
            print(input_time, ' cannot be converted to ', train_time, ' as it is smaller')

        # if(train_period=='week'):
        if train_period == 'week':
            data = self.resemble_week(data_info[self.name])
        data = data[self.name].resample(self.time[train_period]).sum()
        if pct_require:
            data = self.percent_change(data, train_period).dropna()
        train = self.outlier(data, lower=lo, upper=hi)
        train = train.interpolate(method='cubic', order=3, s=0).ffill().bfill()
        model = Holt(train)
        model_fit = model.fit()
        future_forecast = model_fit.predict(len(train), len(train) + self.no_of_forecast_month - 1).reset_index()
        future_forecast = future_forecast.rename(columns={'index': 'Date', 0: self.name})
        train = train.reset_index()
        train = train.append(future_forecast)
        # train = self.percent_change(train).reset_index()
        # train = pd.melt(train, id_vars=[train.columns[0]], value_vars=[train.columns[1]])

        train = train[len(train) - self.no_of_forecast_month:len(train)].reset_index(drop=True)
        time_add = train_period + 's'
        if time_add == 'quarters':
            last_date = train['Date'][len(train) - 1] + relativedelta(months=4)
        else:
            last_date = train['Date'][len(train) - 1] + relativedelta(**{time_add: 1})
        dummy_data = pd.DataFrame(columns=['Date', self.name])
        dummy_data.loc[0] = [last_date, 0]
        train = train.append(dummy_data)
        train = train.set_index('Date')
        train_converted = train[self.name].resample(self.time[input_period], fill_method='ffill')
        train_converted = train_converted[:-1].reset_index()
        portion = percent_portion
        if self.frequency[input_time[0]] == self.frequency[train_time[0]]:
            portion[self.name] = 1
        train_converted['Date'] = pd.to_datetime(train_converted['Date'])
        train_converted['month'] = pd.DatetimeIndex(train_converted['Date']).month
        join_data = train_converted.merge(percent_portion, left_on=input_period, right_on=input_period)
        join_data[self.name] = join_data[self.name + '_x'] * join_data[self.name + '_y']
        join_data = join_data.sort_values(by=['Date']).set_index('Date')
        forecast_value = pd.Series(join_data[self.name], index=join_data.index)
        final = forecast_value.resample(self.time[output_period]).sum().reset_index()
        final = pd.melt(final, id_vars=[final.columns[0]], value_vars=[final.columns[1]])
        print(final)

    def arima_to_fill(self,data, last_time, total_time, period):
        model = pm.auto_arima(data, m=1, start_p=0,start_q=0
                                   , max_p=5, max_q=5, d=2, D=1, seasonal=True
                                   , start_P=0, max_P=5, start_Q=0, max_Q=5
                                   , error_action='ignore', suppress_warnings=True
                                   , stepwise=True)
        future_forecast = model.predict(n_periods=int(total_time-last_time))
        future_date = [j for j in
                       pd.date_range(data.index[len(data) - 1], periods=int(total_time-last_time)+1,
                                     freq=self.time[period])]

        # store future_forcast
        future_forecast = pd.DataFrame(future_forecast, index=future_date[1:],
                              columns=[self.data.columns[self.column_no - 1]])
        data = data.append(future_forecast)
        return data

    def remove(self,data, last_time):
        data = data.drop(data.tail(last_time).index)
        return data

    def hwa(self, hi, lo,input_period, train_period, output_period, pct_require):

        input_time = self.extract_time_type(input_period)
        train_time = self.extract_time_type(train_period)

        # extract column from data using column_no
        data = self.data.iloc[:, self.column_no-1].dropna()
        data = pd.DataFrame(data)
        # convert from month to year
        if self.frequency[input_time[0]] >= self.frequency[train_time[0]]:
            data_check = data.copy()
            if train_period == 'week':
                previous_period = 'week'
            if train_period == 'month':
                previous_period = 'day'
            if train_period == 'quarter':
                previous_period = 'quarter'
            if train_period == 'year':
                previous_period = 'month'
            last_time = getattr(data.index[len(data) - 1], previous_period)
            # cutting data if it's less than 6
            if last_time > self.frequency[input_period] / 2:
                data = self.arima_to_fill(data, last_time, self.frequency[previous_period], input_period)
            else:
                data = self.remove(data, last_time)
            data_info = self.data_info(data)
            # get portion
            if train_period == 'month' or train_period == 'year':
                percent_portion = self.percentage_type1(data_info, input_time[0], train_time[0]).reset_index()
            else:
                percent_portion = self.percentage_type2(data_info, input_time[0], train_time[0]).reset_index()
        else:
            print(input_time, ' cannot be converted to ', train_time, ' as it is smaller')

        # if(train_period=='week'):
        if train_period == 'week':
            data = self.resemble_week(data_info[self.name])
        data = data[self.name].resample(self.time[train_period]).sum()
        if pct_require:
            data = self.percent_change(data, train_period).dropna()
        train = self.outlier(data, lower=lo, upper=hi)
        train = train.interpolate(method='cubic').ffill().bfill()
        model = ExponentialSmoothing(train)#, seasonal_periods=self.frequency[train_period], trend='add', seasonal='add')
        model_fit = model.fit()
        future_forecast = model_fit.predict(len(train), len(train) + self.no_of_forecast_month - 1).reset_index()
        future_forecast = future_forecast.rename(columns={'index': 'Date', 0: self.name})
        train = train.reset_index()
        train = train.append(future_forecast)
        train = train[len(train) - self.no_of_forecast_month:len(train)].reset_index(drop=True)
        time_add = train_period + 's'
        if time_add == 'quarters':
            last_date = train['Date'][len(train) - 1] + relativedelta(months=4)
        else:
            last_date = train['Date'][len(train) - 1] + relativedelta(**{time_add: 1})
        dummy_data = pd.DataFrame(columns=['Date', self.name])
        dummy_data.loc[0] = [last_date, 0]
        train = train.append(dummy_data)
        train = train.set_index('Date')
        train_converted = train[self.name].resample(self.time[input_period], fill_method='ffill')
        train_converted = train_converted[:-1].reset_index()
        portion = percent_portion
        if self.frequency[input_time[0]] == self.frequency[train_time[0]]:
            portion[self.name] = 1
        train_converted['Date'] = pd.to_datetime(train_converted['Date'])
        train_converted['month'] = pd.DatetimeIndex(train_converted['Date']).month
        join_data = train_converted.merge(percent_portion, left_on=input_period, right_on=input_period)
        join_data[self.name] = join_data[self.name+'_x']*join_data[self.name+'_y']
        join_data = join_data.sort_values(by=['Date']).set_index('Date')
        forecast_value = pd.Series(join_data[self.name], index=join_data.index)
        if output_period == 'week':
            forecast_value = self.data_info(forecast_value)
            forecast_value = self.resemble_week(data_info)
        final = forecast_value.resample(self.time[output_period]).sum().reset_index()
        final = pd.melt(final, id_vars=[final.columns[0]], value_vars=[final.columns[1]])
        final['month'] = pd.DatetimeIndex(final['Date']).month
        final['day'] = pd.DatetimeIndex(final['Date']).day
        final['year'] = pd.DatetimeIndex(final['Date']).year
        print(final)

    # multiple holtwinter
    def hwm(self, hi, lo, input_period, train_period, output_period, pct_require):
        input_time = self.extract_time_type(input_period)
        train_time = self.extract_time_type(train_period)

        # extract column from data using column_no
        data = self.data.iloc[:, self.column_no - 1].dropna()
        data = pd.DataFrame(data)
        # convert from month to year
        if self.frequency[input_time[0]] >= self.frequency[train_time[0]]:
            data_check = data.copy()
            if train_period == 'week':
                previous_period = 'week'
            if train_period == 'month':
                previous_period = 'day'
            if train_period == 'quarter':
                previous_period = 'quarter'
            if train_period == 'year':
                previous_period = 'month'
            last_time = getattr(data.index[len(data) - 1], previous_period)
            # cutting data if it's less than 6
            if last_time > self.frequency[input_period] / 2:
                data = self.arima_to_fill(data, last_time, self.frequency[previous_period], input_period)
            else:
                data = self.remove(data, last_time)
            data_info = self.data_info(data)
            # get portion
            if train_period == 'month' or train_period == 'year':
                percent_portion = self.percentage_type1(data_info, input_time[0], train_time[0]).reset_index()
            else:
                percent_portion = self.percentage_type2(data_info, input_time[0], train_time[0]).reset_index()
        else:
            print(input_time, ' cannot be converted to ', train_time, ' as it is smaller')

        # if(train_period=='week'):
        if train_period == 'week':
            data = self.resemble_week(data_info[self.name])
        data = data[self.name].resample(self.time[train_period]).sum()
        if pct_require:
            data = self.percent_change(data, train_period).dropna()
        train = self.outlier(data, lower=lo, upper=hi)
        train = train.interpolate(method='cubic', order=3, s=0).ffill().bfill()
        model = ExponentialSmoothing(train)#, seasonal_periods=self.frequency[train_period], trend='add', seasonal='mul')
        model_fit = model.fit()
        future_forecast = model_fit.predict(len(train), len(train) + self.no_of_forecast_month - 1).reset_index()
        future_forecast = future_forecast.rename(columns={'index': 'Date', 0:self.name})

        train = train.reset_index()
        train = train.append(future_forecast)

        train = train[len(train) - self.no_of_forecast_month:len(train)].reset_index(drop=True)
        time_add = train_period + 's'
        if time_add == 'quarters':
            last_date = train['Date'][len(train) - 1] + relativedelta(months=4)
        else:
            last_date = train['Date'][len(train) - 1] + relativedelta(**{time_add: 1})
        dummy_data = pd.DataFrame(columns=['Date', self.name])
        dummy_data.loc[0] = [last_date, 0]
        train = train.append(dummy_data)
        train = train.set_index('Date')
        train_converted = train[self.name].resample(self.time[input_period], fill_method='ffill')
        train_converted = train_converted[:-1].reset_index()
        portion = percent_portion
        if self.frequency[input_time[0]] == self.frequency[train_time[0]]:
            portion[self.name] = 1
        train_converted['Date'] = pd.to_datetime(train_converted['Date'])
        train_converted['month'] = pd.DatetimeIndex(train_converted['Date']).month
        join_data = train_converted.merge(percent_portion, left_on=input_period, right_on=input_period)
        join_data[self.name] = join_data[self.name + '_x'] * join_data[self.name + '_y']
        join_data = join_data.sort_values(by=['Date']).set_index('Date')
        forecast_value = pd.Series(join_data[self.name], index=join_data.index)
        final = forecast_value.resample(self.time[output_period]).sum().reset_index()
        final = pd.melt(final, id_vars=[final.columns[0]], value_vars=[final.columns[1]])
        print(final)

    # auto arima and find mape
    def arima(self, hi, lo, p, q, P, Q, input_period, train_period, output_period, pct_require):
        input_time = self.extract_time_type(input_period)
        train_time = self.extract_time_type(train_period)

        # extract column from data using column_no
        data = self.data.iloc[:, self.column_no - 1].dropna()
        data = pd.DataFrame(data)
        # convert from month to year
        if self.frequency[input_time[0]] >= self.frequency[train_time[0]]:
            data_check = data.copy()
            if train_period == 'week':
                previous_period = 'week'
            if train_period == 'month':
                previous_period = 'day'
            if train_period == 'quarter':
                previous_period = 'quarter'
            if train_period == 'year':
                previous_period = 'month'
            last_time = getattr(data.index[len(data) - 1], previous_period)
            # cutting data if it's less than 6
            if last_time > self.frequency[input_period] / 2:
                data = self.arima_to_fill(data, last_time, self.frequency[previous_period], input_period)
            else:
                data = self.remove(data, last_time)
            data_info = self.data_info(data)
            # get portion
            if train_period == 'month' or train_period == 'year':
                percent_portion = self.percentage_type1(data_info, input_time[0], train_time[0]).reset_index()
            else:
                percent_portion = self.percentage_type2(data_info, input_time[0], train_time[0]).reset_index()
        else:
            print(input_time, ' cannot be converted to ', train_time, ' as it is smaller')

        # if(train_period=='week'):
        if train_period == 'week':
            data = self.resemble_week(data_info[self.name])
        data = data[self.name].resample(self.time[train_period]).sum()
        if pct_require:
            data = self.percent_change(data, train_period).dropna()
        train = self.outlier(data, lower=lo, upper=hi)
        train = train.interpolate(method='cubic').ffill().bfill()
        train.index = pd.to_datetime(train.index)

        model = pm.auto_arima(train, m=1, start_p=0,start_q=0
                                   , max_p=p, max_q=q, d=2, D=1, seasonal=True
                                   , start_P=0, max_P=P, start_Q=0, max_Q=Q
                                   , error_action='ignore', suppress_warnings=True
                                   , stepwise=True)

        # predict
        future_forecast = model.predict(n_periods=self.no_of_forecast_month)
        future_date = [j for j in
                       pd.date_range(train.index[len(train) - 1], periods=self.no_of_forecast_month + 1, freq=self.time[train_period])]

        # store future_forcast
        future_forecast = pd.DataFrame(future_forecast, index=future_date[1:], columns=[self.data.columns[self.column_no - 1]])
        train = future_forecast.copy().reset_index()
        train = train.rename(columns={'index': 'Date'})
        time_add = train_period + 's'
        if time_add == 'quarters':
            last_date = train['Date'][len(train) - 1] + relativedelta(months=4)
        else:
            last_date = train['Date'][len(train) - 1] + relativedelta(**{time_add: 1})
        dummy_data = pd.DataFrame(columns=['Date', self.name])
        dummy_data.loc[0] = [last_date, 0]
        train = train.append(dummy_data)
        train = train.set_index('Date')

        train_converted = train[self.name].resample(self.time[input_period], fill_method='ffill')
        train_converted = train_converted[:-1].reset_index()
        portion = percent_portion
        if self.frequency[input_time[0]] == self.frequency[train_time[0]]:
            portion[self.name] = 1
        train_converted['Date'] = pd.to_datetime(train_converted['Date'])
        train_converted['month'] = pd.DatetimeIndex(train_converted['Date']).month
        join_data = train_converted.merge(percent_portion, left_on=input_period, right_on=input_period)
        join_data[self.name] = join_data[self.name + '_x'] * join_data[self.name + '_y']
        join_data = join_data.sort_values(by=['Date']).set_index('Date')
        forecast_value = pd.Series(join_data[self.name], index=join_data.index)
        final = forecast_value.resample(self.time[output_period]).sum().reset_index()
        final = pd.melt(final, id_vars=[final.columns[0]], value_vars=[final.columns[1]])
        print(final)

    # leading is not done yet only except leading with month
    def arimax(self, hi, lo, p, q, P, Q, leading_name, lag, trend_require, percentchange_require, input_period
               , train_period, output_period, pct_require):

        f_data = pd.read_csv(r'C:\Users\mankk\PycharmProjects\data_science\venv\leading_arimax.csv', index_col=0)
        input_time = self.extract_time_type(input_period)
        train_time = self.extract_time_type(train_period)
        # extract column from data using column_no
        data = self.data.iloc[:, self.column_no - 1].dropna()
        data = pd.DataFrame(data)
        # convert from month to year
        if self.frequency[input_time[0]] >= self.frequency[train_time[0]]:
            data_check = data.copy()
            if train_period == 'week':
                previous_period = 'week'
            if train_period == 'month':
                previous_period = 'day'
            if train_period == 'quarter':
                previous_period = 'quarter'
            if train_period == 'year':
                previous_period = 'month'
            last_time = getattr(data.index[len(data) - 1], previous_period)
            # cutting data if it's less than 6
            if last_time > self.frequency[input_period] / 2:
                data = self.arima_to_fill(data, last_time, self.frequency[previous_period], input_period)
            else:
                data = self.remove(data, last_time)
            data_info = self.data_info(data)
            # get portion
            if train_period == 'month' or train_period == 'year':
                percent_portion = self.percentage_type1(data_info, input_time[0], train_time[0]).reset_index()
            else:
                percent_portion = self.percentage_type2(data_info, input_time[0], train_time[0]).reset_index()
        else:
            print(input_time, ' cannot be converted to ', train_time, ' as it is smaller')

        # if(train_period=='week'):
        if train_period == 'week':
            data = self.resemble_week(data_info[self.name])
        data = data[self.name].resample(self.time[train_period]).sum()
        if pct_require:
            data = self.percent_change(data, train_period).dropna()
        train = self.outlier(data, lower=lo, upper=hi)
        train.index = pd.to_datetime(train.index)

        # fill outlier with interpolate
        train = train.interpolate(method='cubic', order=3, s=0).ffill().bfill()
        # convert train to percent change
        if pct_require:
            train = self.percent_change(train,train_period).dropna()
        # convert time index to format datetime
        leading_data = f_data.loc[:, leading_name].dropna()
        leading_data.index = pd.to_datetime(leading_data.index)

        # if trend is needed
        if trend_require:
            trend = self.x13(leading_data)
            leading_data = trend.trend

        # if percent year change is needed
        if percentchange_require:
            leading_data = self.percent_change(leading_data,train_period)
        # adding null value to date frame in order to shift
        start_train_date = train.index[0]
        # leading_data.index คือ datetime
        # pd.offsets.MonthOffset(lag) บวกไป lag เดือน
        leading_data.index = leading_data.index+pd.offsets.MonthOffset(1)

        end_train_date = train.index[len(train) - 1]
        leading_data = leading_data[start_train_date:end_train_date]

        leading = leading_data.values
        if percentchange_require:
            leading = leading.dot(100)

        # reshape array to fir auto arima in shape of (-1,1)
        leading = np.array(leading).reshape(-1, 1)

        # start_q is for start ar || start q is for start ma ||and max is for max value
        # start_Q is for start ar for seasonal || start_Q is for ma for start seasonal
        model = pm.auto_arima(train, d=2, D=2, exogenous=leading,
                              start_p=0, start_q=0, max_p=p, max_q=q, m=365,
                              seasonal=True, trace=False, error_action='ignore',
                              start_P=0, start_Q=0, max_P=P, max_Q=Q,
                              suppress_warnings=True, stepwise=True,stationary=True)

        # predict
        future_forecast = model.predict(n_periods=self.no_of_forecast_month
                                        , exogenous=leading[len(leading)-4:len(leading)])
        print(future_forecast)
        # get list of datetime from the end date of train
        future_date = [j for j in
                       pd.date_range(train.index[len(train) - 1], periods=self.no_of_forecast_month + 1,
                                     freq=self.time[train_period])]

        # store future_forcast
        future_forecast = pd.DataFrame(future_forecast, index=future_date[1:],
                                       columns=[self.data.columns[self.column_no - 1]])
        # train = train.append(future_forecast).reset_index()

        # train = self.percent_change(train).reset_index()
        # train = train.rename(columns={'index': 'Date'})
        # train = train[len(train)-self.no_of_forecast_month:len(train)].reset_index(drop=True)
        train = future_forecast.copy().reset_index()
        train = train.rename(columns={'index': 'Date'})
        time_add = train_period + 's'
        if time_add == 'quarters':
            last_date = train['Date'][len(train) - 1] + relativedelta(months=4)
        else:
            last_date = train['Date'][len(train) - 1] + relativedelta(**{time_add: 1})
        dummy_data = pd.DataFrame(columns=['Date', self.name])
        dummy_data.loc[0] = [last_date, 0]
        train = train.append(dummy_data)
        train = train.set_index('Date')

        train_converted = train[self.name].resample(self.time[input_period], fill_method='ffill')
        train_converted = train_converted[:-1].reset_index()
        portion = percent_portion
        if self.frequency[input_time[0]] == self.frequency[train_time[0]]:
            portion[self.name] = 1
        train_converted['Date'] = pd.to_datetime(train_converted['Date'])
        train_converted['month'] = pd.DatetimeIndex(train_converted['Date']).month
        join_data = train_converted.merge(percent_portion, left_on=input_period, right_on=input_period)
        join_data[self.name] = join_data[self.name + '_x'] * join_data[self.name + '_y']
        join_data = join_data.sort_values(by=['Date']).set_index('Date')
        forecast_value = pd.Series(join_data[self.name], index=join_data.index)
        final = forecast_value.resample(self.time[output_period]).sum().reset_index()
        final = pd.melt(final, id_vars=[final.columns[0]], value_vars=[final.columns[1]])
        print(final)





