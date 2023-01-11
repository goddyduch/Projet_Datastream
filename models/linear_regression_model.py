from river import linear_model

class  ONLINE_REGRESSION():
    def __init__(self):

        self.model = linear_model.BayesianLinearRegression()
        self.first_iterate = True
        self.past_x = None
        self.past_predict = 0

    def learn_one(self, x_future, y_to_pred):
        if self.first_iterate:
            self.first_iterate = False
        else:
            self.model.learn_one(self.past_x,y_to_pred)
        self.past_x = x_future

            
    def predict_one(self,x):
        return self.model.predict_one(x)