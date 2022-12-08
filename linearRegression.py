import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class LRegression(object):
    def __init__(self, data_type=None, test=False, pred=False, direct=True):
        self.data_type = data_type
        self.direct = direct
        self.load_data()
        
        if test:
            self.fit_model()
        elif pred:
            self.load_model()
            self.test_model()
            self.plot_predictions()
            # self.predictions = {}
            self.save_predictions()
        else:
            self.fit_model()
            self.test_model()
            self.plot_predictions()
            self.save_predictions()
        
    
    def load_data(self):
        data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/data/modeling_data/{}/'.format(self.data_type)
        self.train_df = pd.read_csv(data_path + 'train_wpreds.txt')
        # print(self.train_df['Orgn_Drgn'].unique())
        # exit()
        self.test_df = pd.read_csv(data_path + 'test_wpreds.txt')
        with open(data_path + 'scale_parameters.pickle', 'rb') as handle:
            scale_parameters = pickle.load(handle)
        
        self.variables_means = scale_parameters['means']
        self.variables_stds = scale_parameters['stds']
        if self.direct:
            self.X_train = self.train_df[['shortPath_mile', 'adverse_weather','highway','path_avg_bwness','path_avg_degree','cos_theta','sin_theta']].values
            self.X_test = self.test_df[['shortPath_mile', 'adverse_weather','highway','path_avg_bwness','path_avg_degree','cos_theta','sin_theta']].values
            # self.Y_train = self.train_df['unitDist_TT_hpm'].values
            # self.Y_test = self.test_df['unitDist_TT_hpm'].values
            self.Y_train = self.train_df['travel_time'].values
            self.Y_test = self.test_df['travel_time'].values
        else:
            self.X_train = self.train_df[['adverse_weather','highway','path_avg_bwness','path_avg_degree','cos_theta','sin_theta']].values
            self.X_test = self.test_df[['adverse_weather','highway','path_avg_bwness','path_avg_degree','cos_theta','sin_theta']].values
            self.Y_train = self.train_df['unitDist_TT_hpm'].values
            self.Y_test = self.test_df['unitDist_TT_hpm'].values
    
    def fit_model(self):
        self.LR = LinearRegression().fit(self.X_train, self.Y_train)
        save_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/trained_models/{}/LinearRegression/'.format(self.data_type)
        if self.direct:
            with open(save_path + 'model1_MN_LRegression_direct.pickle', 'wb') as buff:
                pickle.dump(self.LR, buff)
        else:
            with open(save_path + 'model1_MN_LRegression.pickle', 'wb') as buff:
                pickle.dump(self.LR, buff)
    
    def load_model(self):
        save_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/trained_models/{}/LinearRegression/'.format(self.data_type)
        if self.direct:
            with open(save_path + 'model1_MN_LRegression_direct.pickle', 'rb') as buff:
                self.LR = pickle.load(buff)
        else:
            with open(save_path + 'model1_MN_LRegression.pickle', 'rb') as buff:
                self.LR = pickle.load(buff)
    
    def mean_relative_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        MRE = np.mean(np.abs((y_true - y_pred) / y_true)) 
        # print('mean relative error is {:0.4f}'.format(MRE) )
        return MRE
    def mean_absolute_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        MAE = np.mean(np.abs(y_true - y_pred)) #* 60
        # print('mean absolute error is {:0.2f} '.format( MAE) )
        return MAE
    def test_model(self):

        self.Y_preds_tst = self.LR.predict(self.X_test)
        # self.Y_dists_tst = self.LR.pred_dist(self.X_test) #.params = {'loc':, 'scale':}
        # self.std_preds_tst = self.Y_dists_tst.params['scale']
        # print(self.std_preds_tst)
        # print(self.std_preds_tst.shape)
        # exit()
        self.test_MSE = mean_squared_error(self.Y_preds_tst, self.Y_test)
        print('Test MSE', np.sqrt(self.test_MSE))
        print("Test mean relative error {:0.4f}".format(self.mean_relative_error(self.Y_test, self.Y_preds_tst)))
        print("Test mean absolute error {:0.2f}".format(self.mean_absolute_error(self.Y_test, self.Y_preds_tst)))
        
        # test Negative Log Likelihood
        # self.test_NLL = -self.Y_dists_tst.logpdf(self.Y_test).mean()
        # print('Test NLL', self.test_NLL)

        self.Y_preds_tr = self.LR.predict(self.X_train)
        # self.Y_dists_tr = self.ngb.pred_dist(self.X_train)
        # self.std_preds_tr = self.Y_dists_tr.params['scale']

        # test Mean Squared Error
        self.train_MSE = mean_squared_error(self.Y_preds_tr, self.Y_train)
        print('Train MSE', np.sqrt(self.train_MSE))
        print("Train mean relative error {:0.4f}".format(self.mean_relative_error(self.Y_train, self.Y_preds_tr)))
        print("Train mean absolute error {:0.2f}".format(self.mean_absolute_error(self.Y_train, self.Y_preds_tr)))
        # # test Negative Log Likelihood
        # self.train_NLL = -self.Y_dists_tr.logpdf(self.Y_train).mean()
        # print('Train NLL', self.train_NLL)

    def plot_predictions(self):
        nplot = 10000
        fig, ax = plt.subplots(ncols = 2, figsize=(10,5))
        # ax[0].plot(self.Y_train[:nplot], self.Y_preds_tr[:nplot],'.' )
        ax[0].plot(self.Y_train[:nplot], self.Y_preds_tr[:nplot], 'r.', alpha=0.4)
        ax[0].plot([0,1000],[0,1000])
        ax[0].set_xlabel('True')
        ax[0].set_ylabel('Pred')
        # ax[1].plot(self.Y_test[:nplot], self.Y_preds_tst[:nplot],'.')
        ax[1].plot(self.Y_test[:nplot], self.Y_preds_tst[:nplot], 'r.', alpha=0.4)
        ax[1].plot([0,1000],[0,1000])
        ax[1].set_xlabel('True')
        ax[1].set_ylabel('Pred')
        plt.show()
    
    def save_predictions(self):

        if self.direct:
            self.train_df['LRegression_tt_predMean_direct'] = self.Y_preds_tr/60.
            # self.train_df['LRegression_tt_predStd_direct'] = self.std_preds_tr/60.

            self.test_df['LRegression_tt_predMean_direct'] = self.Y_preds_tst/60.
            # self.test_df['LRegression_tt_predStd_direct'] = self.std_preds_tst/60.

        else:
            pass
            # self.train_df['LRegression_udtt_predMean'] = self.Y_preds_tr
            # self.train_df['LRegression_udtt_predStd'] = self.std_preds_tr

            # self.test_df['LRegression_udtt_predMean'] = self.Y_preds_tst
            # self.test_df['LRegression_udtt_predStd'] = self.std_preds_tst

            # miles = self.train_df['shortPath_mile'].values
            # self.train_df['LRegression_tt_predMean'] = self.train_df.ngboost_udtt_predMean[:] * miles * 60
            # self.train_df['LRegression_tt_predStd'] = self.train_df.ngboost_udtt_predStd * np.sqrt( miles) * 60

            # miles = self.test_df['shortPath_mile'].values
            # self.test_df['ngboost_tt_predMean'] = self.test_df.ngboost_udtt_predMean * miles * 60
            # self.test_df['ngboost_tt_predStd'] = self.test_df.ngboost_udtt_predStd * np.sqrt( miles) * 60
            

        data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/data/modeling_data/{}/'.format(self.data_type)
        self.train_df.to_csv(data_path + 'train_wpreds.txt')
        self.test_df.to_csv(data_path + 'test_wpreds.txt')