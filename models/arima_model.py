from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings

class  ARIMA_WINDOW():
    def __init__(self, k, d, p, window_size):

        self.k = k
        self.d = d
        self.p = p
        self.history = []
        self.window_size = window_size
        self.model_trained = False
        self.fitted_model = None

    def learn_one(self, y, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.history.append(x)
            if len(self.history) == self.window_size:
                self.model_trained = True
                model = ARIMA(np.array(self.history),
                                            order=(self.k,self.d,self.d))
                model.initialize_approximate_diffuse()
                self.fitted_model = model.fit()
                
            elif len(self.history) > self.window_size:
                self.history.pop(0)
                model = ARIMA(np.array(self.history),
                                            order=(self.k,self.d,self.d))
                model.initialize_approximate_diffuse()
                self.fitted_model = model.fit()
            
    def predict_one(self,x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.model_trained:
                return self.fitted_model.forecast()[0]
            else:
                raise Exception

