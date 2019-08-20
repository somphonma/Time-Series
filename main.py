import re
import math
from Time_Series_Forecasting import *
def main(masterfile_location,datafile_location):

    # instruction_file = r'C:\Users\mankk\PycharmProjects\data_science\venv\master_file.xlsx'
    data_file = r'C:\Users\mankk\PycharmProjects\data_science\venv\Data_testing.csv'
    df = pd.read_excel(masterfile_location)
    ds = pd.read_csv(datafile_location)
    # iteration = loop value
    frame = pd.DataFrame()
    for iteration_value in range(42, len(df)):
        print(df['MAT'][iteration_value])
        lag = np.nan
        leading_name = ''
        mat = df['MAT'][iteration_value].replace(" ", "")
        col_index = -1
        for i in range(0, len(ds)+1):
            k = ds.columns[i].replace(" ", "")
            if k == mat:
                col_index = i
                break
        forecast = 4
        high = df['high'][iteration_value]
        low = df['low'][iteration_value]
        technic = df['technic'][iteration_value]
        dummy = str(leading_name)
        class_forecast = Forecasting(datafile_location, col_index, forecast)
        p = df['max_ar'][iteration_value]
        q = df['max_ma'][iteration_value]
        P = df['max_sar'][iteration_value]
        Q = df['max_sma'][iteration_value]
        trend_require = df['trend'][iteration_value]
        growth = df['growth'][iteration_value]
        k = df['lag'][iteration_value]
        if math.isnan(df['lag'][iteration_value]) is False:
            lag = int(df['lag'][iteration_value])
        if pd.isnull(df['leading'][iteration_value]) is False:
            leading_name = df['leading'][iteration_value]
        input_period = df['input_period'][iteration_value]
        train_period = df['train_period'][iteration_value]
        output_period = df['train_period'][iteration_value]
        print(iteration_value)
        if technic == 'arimax':
            a = 10
            print('t')
            # a = getattr(class_forecast, technic)(hi=high, lo=low, p=p, q=q, P=P, Q=Q, leading_name=leading_name, lag=lag, trend_require=trend_require
            #                                     , percentchange_require=growth, input_period='month', train_period='month', output_period='month', pct_require=True)
        elif technic == 'arima':
            a = getattr(class_forecast, technic)(hi=high, lo=low, p=p, q=q, P=P, Q=Q, input_period='month', train_period='month', output_period='month', pct_require=True)
        else:
            a = getattr(class_forecast, technic)(hi=high, lo=low, input_period=input_period, train_period=train_period, output_period=output_period, pct_require=True)
        if a!=10:
            frame = frame.append(a)
        break


if __name__ == "__main__":
    a = r'C:\Users\mankk\PycharmProjects\data_science\venv\master_file.xlsx'
    b = r'C:\Users\mankk\PycharmProjects\data_science\venv\Data_testing.csv'
    main(a,b)
