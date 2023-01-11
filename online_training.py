from models.arima_model import ARIMA_WINDOW
from river import linear_model
from river.utils import dict2numpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from river import preprocessing

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.techindicators import TechIndicators
from river.stream import iter_pandas
import tqdm

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


class  Online_trainer():
    def __init__(self, key_to_predict, window_size, arima_args=(5,3,5), online_reg_args=(), batch_reg_args=()):

        arima_args_ = arima_args + (window_size,)
        self.arima_model = ARIMA_WINDOW(*arima_args_)
        self.online_regressor = linear_model.LinearRegression()
        self.reach_window = False
        self.n_iterate = 0
        self.previous_x = None
        self.scaler = preprocessing.AdaptiveStandardScaler(alpha=.6)
        # self.scaler = preprocessing.StandardScaler()
        self.key_to_predict = key_to_predict

    def learn_one(self, x_y):
        self.n_iterate += 1
        if self.n_iterate == 100:
            self.reach_window = True
        
        input = self.scaler.learn_one(x_y).transform_one(x_y)
        label = x_y[self.key_to_predict]

        ###online_regressor
        self.online_regressor.learn_one(input,label)

        ####arima
        self.arima_model.learn_one(input,label)

            
    def predict_one(self, x_y):
        if self.reach_window:
            input = self.scaler.transform_one(x_y)
            label = x_y[self.key_to_predict]

            result_online_reg = self.online_regressor.predict_one(input)
            result_arima_reg = self.arima_model.predict_one(input)

            return label, result_online_reg, result_arima_reg
        else:
            raise ValueError


def training_loop(stream, trainer, len):

    y_true = []
    y_pred_online_reg = []
    y_pred_online_arima = []
    
    for i, x_ in enumerate(stream):
        x_y = x_[0]

        if i >= 100:
            print(i)
            label, y_or, y_ar = trainer.predict_one(x_y)
            print(label, y_ar)
            y_true.append(label)
            y_pred_online_reg.append(y_or)
            y_pred_online_arima.append(y_ar)
        trainer.learn_one(x_y)

    plt.plot(y_true)
    # plt.plot(y_pred_online_reg)
    plt.plot(y_pred_online_arima)
    plt.axis([0, len, min(y_true)- 100, max(y_true)+ 100])
    plt.show()

"""----------------------------------IMPORTATION DES VALEURS BOURSIERES --------------------------------"""
key = '8NY5A51OROX11O2Y'
ts = TimeSeries(key=key,output_format='pandas', indexing_type='date')


Nasdaq = 'NDAQ'
sp_500 = 'SPY'
CAC_40 = 'PX1'

#IMPORTATION NASDAQ
ts = TimeSeries(key = key, output_format = 'csv')
totalData = ts.get_intraday_extended(symbol ='SPY', interval = '15min', slice = 'year1month1')
df = pd.DataFrame(list(totalData[0]))

#setup of column and index
header_row=0
df.columns = df.iloc[header_row]
df = df.drop(df.index[0])
df = df.iloc[::-1]

#day of week
df['time'] = pd.to_datetime(df['time'])
df = pd.concat((df, pd.get_dummies(df['time'].dt.day_name()).astype(float)), axis=1)
df = df.set_index('time')
df.loc[:, df.columns != 'day_of_week'] = df.loc[:, df.columns != 'day_of_week'].apply(pd.to_numeric)
df['volume'] = df['volume'].astype(float)


x_y = df.copy()

trainer = Online_trainer(key_to_predict='open',window_size = 100)
training_loop(iter_pandas(x_y), trainer, len=x_y.shape[0])