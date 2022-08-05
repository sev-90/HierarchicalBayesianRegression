import numpy as np
import pickle
import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# import arviz as az

class models_comparison(object):
    def __init__(self, data_type = None ):
        self.data_type = data_type
        self.load_data()
        # self.udtt_plot_predictions()
        # self.tt_predictions_plot()
        self.print_error()
        # self.plot_predictions_byRegions()
        # self.load_models()
        # self.test_model()
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
        # print(self.test_df)
        # exit()


    def udtt_plot_predictions(self):
        nplot = 100
        fig, ax = plt.subplots(ncols = 2, figsize=(10,5))
        # ax[0].plot(self.train_df.unitDist_TT_hpm[:], self.train_df.hbr_predMean_onTrain[:],'b.' , markersize=0.5)
        # ax[0].plot(self.train_df.unitDist_TT_hpm[:], self.train_df.ngboost_predMean_onTrain[:],'r.' ,markersize=0.5)
        ax[0].errorbar(x = self.train_df.unitDist_TT_hpm[:], y = self.train_df.hbr_udtt_predMean[:], yerr= self.train_df.hbr_udtt_predStd[:], fmt='b.',markersize=0.5, alpha=0.2)
        ax[0].errorbar(x =self.train_df.unitDist_TT_hpm[:], y = self.train_df.ngboost_udtt_predMean[:], yerr= self.train_df.ngboost_udtt_predStd[:], fmt='r.', markersize=0.5,alpha=0.2)
        ax[0].plot([0,0.25],[0,0.25])
        ax[0].set_xlabel('True')
        ax[0].set_ylabel('Pred')
        # ax[1].plot(self.test_df.unitDist_TT_hpm[:], self.test_df.hbr_predMean_onTest[:],'b.',markersize=0.5 )
        # ax[1].plot(self.test_df.unitDist_TT_hpm[:], self.test_df.ngboost_predMean_onTest[:],'r.',markersize=0.5 )
        ax[1].errorbar(x = self.test_df.unitDist_TT_hpm[:], y = self.test_df.hbr_udtt_predMean[:], yerr= self.test_df.hbr_udtt_predStd[:], fmt='b.',markersize=0.5, alpha=0.2)
        ax[1].errorbar(x =self.test_df.unitDist_TT_hpm[:], y = self.test_df.ngboost_udtt_predMean[:], yerr= self.test_df.ngboost_udtt_predStd[:], fmt='r.', markersize=0.5,alpha=0.2)
        ax[1].plot([0,0.25],[0,0.25])
        ax[1].set_xlabel('True')
        ax[1].set_ylabel('Pred')
        plt.show()
    
    
    def tt_predictions_plot(self):

        nplot = 100
        fig, ax = plt.subplots(ncols = 2, figsize=(10,5))
        # ax[0].plot(self.train_df.unitDist_TT_hpm[:], self.train_df.hbr_predMean_onTrain[:],'b.' , markersize=0.5)
        # ax[0].plot(self.train_df.unitDist_TT_hpm[:], self.train_df.ngboost_predMean_onTrain[:],'r.' ,markersize=0.5)
        ax[0].errorbar(x = self.train_df.travel_time[:]/60., y = self.train_df.hbr_tt_predMean[:], yerr= self.train_df.hbr_tt_predStd[:], fmt='b.',markersize=0.5, alpha=0.2)
        ax[0].errorbar(x =self.train_df.travel_time[:]/60., y = self.train_df.ngboost_tt_predMean[:], yerr= self.train_df.ngboost_tt_predStd[:], fmt='r.', markersize=0.5,alpha=0.2)
        ax[0].plot([0,50],[0,50], color='k')
        ax[0].set_xlabel('True')
        ax[0].set_ylabel('Pred')
        # ax[1].plot(self.test_df.unitDist_TT_hpm[:], self.test_df.hbr_predMean_onTest[:],'b.',markersize=0.5 )
        # ax[1].plot(self.test_df.unitDist_TT_hpm[:], self.test_df.ngboost_predMean_onTest[:],'r.',markersize=0.5 )
        ax[1].errorbar(x = self.test_df.travel_time[:]/60., y = self.test_df.hbr_tt_predMean[:], yerr= self.test_df.hbr_tt_predStd[:], fmt='b.',markersize=0.5, alpha=0.2)
        ax[1].errorbar(x =self.test_df.travel_time[:]/60., y = self.test_df.ngboost_tt_predMean[:], yerr= self.test_df.ngboost_tt_predStd[:], fmt='r.', markersize=0.5,alpha=0.2)
        ax[1].plot([0,50],[0,50], color='k')
        ax[1].set_xlabel('True')
        ax[1].set_ylabel('Pred')
        plt.show()
    
    def mean_relative_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        MRE = np.mean(np.abs((y_true - y_pred) / y_true)) 
        print('mean relative error is ', MRE )
        return MRE
    
    def mean_absolute_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        MAE = np.mean(np.abs(y_true - y_pred)) * 60
        print('mean absolute error is ', MAE )
        return MAE
    
    def root_mean_sqr_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        RMSE = np.sqrt(mean_squared_error(y_true, y_pred)) * 60
        print('root mean square error is ', RMSE )
        return RMSE
    
    def median_absolute_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        MedAE = np.median(np.abs(y_true - y_pred)) *60
        print('median absolute error is ', MedAE )
        return MedAE
    
    def median_relative_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        MedRE = np.median(np.abs((y_true - y_pred) / y_true)) 
        print('median relative error is ', MedRE )
        return MedRE


    def print_error(self):
        # print('**** for hbr model the errors on tt are: ')
        print("***for hbr model the errors on tt on train and test:")
        self.mean_relative_error( self.train_df.travel_time/60., self.train_df.hbr_tt_predMean_logn)
        self.mean_relative_error( self.test_df.travel_time/60., self.test_df.hbr_tt_predMean_logn)

        self.mean_absolute_error(self.train_df.travel_time/60., self.train_df.hbr_tt_predMean_logn)
        self.mean_absolute_error(self.test_df.travel_time/60., self.test_df.hbr_tt_predMean_logn)
        
        self.median_relative_error(self.train_df.travel_time/60., self.train_df.hbr_tt_predMean_logn)
        self.median_relative_error(self.test_df.travel_time/60., self.test_df.hbr_tt_predMean_logn)

        self.median_absolute_error(self.train_df.travel_time/60., self.train_df.hbr_tt_predMean_logn)
        self.median_absolute_error(self.test_df.travel_time/60., self.test_df.hbr_tt_predMean_logn)

        self.root_mean_sqr_error(self.train_df.travel_time/60., self.train_df.hbr_tt_predMean_logn)
        self.root_mean_sqr_error(self.test_df.travel_time/60., self.test_df.hbr_tt_predMean_logn)

 
        # print('**** for direct hbr model the errors on tt are: ')
        print("*** for direct hbr model the errors on tt on train and test:")
        self.mean_relative_error( self.train_df.travel_time/60., self.train_df.hbr_tt_predMean_logdirect)
        self.mean_relative_error( self.test_df.travel_time/60., self.test_df.hbr_tt_predMean_logdirect)

        self.mean_absolute_error(self.train_df.travel_time/60., self.train_df.hbr_tt_predMean_logdirect)
        self.mean_absolute_error(self.test_df.travel_time/60., self.test_df.hbr_tt_predMean_logdirect)

        self.median_relative_error(self.train_df.travel_time/60., self.train_df.hbr_tt_predMean_logdirect)
        self.median_relative_error(self.test_df.travel_time/60., self.test_df.hbr_tt_predMean_logdirect)
        
        self.median_absolute_error(self.train_df.travel_time/60., self.train_df.hbr_tt_predMean_logdirect)
        self.median_absolute_error(self.test_df.travel_time/60., self.test_df.hbr_tt_predMean_logdirect)


        self.root_mean_sqr_error(self.train_df.travel_time/60., self.train_df.hbr_tt_predMean_logdirect)
        self.root_mean_sqr_error(self.test_df.travel_time/60., self.test_df.hbr_tt_predMean_logdirect)


        # print('**** for ngboost model the errors on tt are: ')
        print("***for ngboost model the errors on tt on train and test:")
        self.mean_relative_error( self.train_df.travel_time/60., self.train_df.ngboost_tt_predMean)
        self.mean_relative_error( self.test_df.travel_time/60., self.test_df.ngboost_tt_predMean)

        self.mean_absolute_error(self.train_df.travel_time/60., self.train_df.ngboost_tt_predMean)
        self.mean_absolute_error(self.test_df.travel_time/60., self.test_df.ngboost_tt_predMean)

        self.median_relative_error(self.train_df.travel_time/60., self.train_df.ngboost_tt_predMean)
        self.median_relative_error(self.test_df.travel_time/60., self.test_df.ngboost_tt_predMean)
        
        self.median_absolute_error(self.train_df.travel_time/60., self.train_df.ngboost_tt_predMean)
        self.median_absolute_error(self.test_df.travel_time/60., self.test_df.ngboost_tt_predMean)


        self.root_mean_sqr_error(self.train_df.travel_time/60., self.train_df.ngboost_tt_predMean)
        self.root_mean_sqr_error(self.test_df.travel_time/60., self.test_df.ngboost_tt_predMean)

       

        # print('**** for direct ngboost model the errors on tt are: ')
        print("*** for direct ngboost model the errors on tt on train and test:")
        self.mean_relative_error( self.train_df.travel_time/60., self.train_df.ngboost_tt_predMean_direct)
        self.mean_relative_error( self.test_df.travel_time/60., self.test_df.ngboost_tt_predMean_direct)

        self.mean_absolute_error(self.train_df.travel_time/60., self.train_df.ngboost_tt_predMean_direct)
        self.mean_absolute_error(self.test_df.travel_time/60., self.test_df.ngboost_tt_predMean_direct)

        self.median_relative_error(self.train_df.travel_time/60., self.train_df.ngboost_tt_predMean_direct)
        self.median_relative_error(self.test_df.travel_time/60., self.test_df.ngboost_tt_predMean_direct)
        
        self.median_absolute_error(self.train_df.travel_time/60., self.train_df.ngboost_tt_predMean_direct)
        self.median_absolute_error(self.test_df.travel_time/60., self.test_df.ngboost_tt_predMean_direct)


        self.root_mean_sqr_error(self.train_df.travel_time/60., self.train_df.ngboost_tt_predMean_direct)
        self.root_mean_sqr_error(self.test_df.travel_time/60., self.test_df.ngboost_tt_predMean_direct)
  

        
  

        # # print('**** for hbr model the errors on udtt are: ')
        # # print("*** on train:")
        # # self.mean_relative_error( self.train_df.unitDist_TT_hpm, self.train_df.hbr_udtt_predMean)
        # # self.mean_absolute_error(self.train_df.unitDist_TT_hpm, self.train_df.hbr_udtt_predMean)
        # # print("**** on test:")
        # # self.mean_relative_error( self.test_df.unitDist_TT_hpm, self.test_df.hbr_udtt_predMean)
        # # self.mean_absolute_error(self.test_df.unitDist_TT_hpm, self.test_df.hbr_udtt_predMean)


        # # print('**** for ngboost model the errors on udtt are: ')
        # # print("*** on train:")
        # # self.mean_relative_error( self.train_df.unitDist_TT_hpm, self.train_df.ngboost_udtt_predMean)
        # # self.mean_absolute_error(self.train_df.unitDist_TT_hpm, self.train_df.ngboost_udtt_predMean)
        # # print("**** on test:")
        # # self.mean_relative_error( self.test_df.unitDist_TT_hpm, self.test_df.ngboost_udtt_predMean)
        # # self.mean_absolute_error(self.test_df.unitDist_TT_hpm, self.test_df.ngboost_udtt_predMean)

    def plot_predictions_byRegions(self):
        for pair in self.train_df['Orgn_Drgn'].unique():
            d = self.train_df[self.train_df['Orgn_Drgn']==pair]
            errors =  d['travel_time'][:]/60. - d['hbr_tt_predMean_direct'][:]
            # sq_errors = (d['travel_time'][:]/60. - d['predictions1'][:]/60.)**2

            d_tst = self.test_df[self.test_df['Orgn_Drgn']==pair]
            errors_tst =  d_tst['travel_time'][:]/60. - d_tst['hbr_tt_predMean_direct'][:]

            col = 4
            fig, ax = plt.subplots(ncols = col, figsize=(7* col, 6))

            ax[0].errorbar(x = d['travel_time'][:]/60, y = d['hbr_tt_predMean_direct'][:], yerr= d['hbr_tt_predStd_direct'][:], fmt='.', alpha=0.4, label=pair)
            ax[0].plot([0,50],[0,50], color='k')
            ax[0].set_xlabel('observed (min)', fontsize=22, fontdict={'weight': 'bold'})
            ax[0].set_ylabel('predicted (min)', fontsize=22, fontdict={'weight': 'bold'})
            # ax[0].set_yticklabels(ax[0].get_yticks().astype(int), size = 22, fontdict={'weight': 'bold'})
            # ax[0].set_xticklabels(ax[0].get_xticks().astype(int), size = 22, fontdict={'weight': 'bold'})
            ax[0].tick_params(axis='both', which='major', labelsize=22)
            ax[0].legend(fontsize=22)

            _ = ax[1].hist(errors, bins=50, density=True)
            mean_rmse = np.sqrt(mean_squared_error(d['travel_time'][:]/60, d['hbr_tt_predMean_direct'][:]))
            std_rmse = np.std(errors)
            _=ax[1].set_title(" Train: RMSE = {:.2f} $\pm$ {:.2f} (min)".format(mean_rmse,std_rmse),fontsize=22, fontdict={'weight': 'bold'})
            ax[1].set_xlabel('error(min)', fontsize=22, fontdict={'weight': 'bold'})
            # ax[1].set_yticklabels(ax[1].get_yticks(), size = 22, fontdict={'weight': 'bold'})
            # ax[1].set_xticklabels(np.round(ax[1].get_xticklabels(),1), size = 22, fontdict={'weight': 'bold'})
            ax[1].tick_params(axis='both', which='major', labelsize=22)
            ax[1].set_ylabel('')
            ax[1].set_yticks([])

            ax[2].errorbar(x = d_tst['travel_time'][:]/60., y = d_tst['hbr_tt_predMean_direct'][:], yerr= d_tst['hbr_tt_predStd_direct'][:], fmt='r.', alpha=0.4, label=pair)
            ax[2].plot([0,50],[0,50], color='k')
            ax[2].set_xlabel('observed (min)', fontsize=22, fontdict={'weight': 'bold'})
            # ax[2].set_ylabel('predicted (min)', fontsize=22, fontdict={'weight': 'bold'})
            # ax[2].set_yticklabels(ax[2].get_yticks().astype(int), size = 22, fontdict={'weight': 'bold'})
            # ax[2].set_xticklabels(ax[2].get_xticks().astype(int), size = 22, fontdict={'weight': 'bold'})
            ax[2].tick_params(axis='both', which='major', labelsize=22)
            ax[2].legend(fontsize=22)

            _ = ax[3].hist(errors_tst, bins=50, density=True, color='r')
            # ax[3].set_yticklabels(ax[3].get_yticks(), size = 22, fontdict={'weight': 'bold'})
            # ax[3].set_xticklabels(ax[3].get_xticks(), size = 22, fontdict={'weight': 'bold'})
            ax[3].tick_params(axis='both', which='major', labelsize=22)
            mean_rmse_tst = np.sqrt(mean_squared_error(d_tst['travel_time'][:]/60,d_tst['hbr_tt_predMean_direct'][:]))
            std_rmse_tst = np.std(errors_tst)
            _=ax[3].set_title(" Test: RMSE = {:.2f} $\pm$ {:.2f} (min)".format(mean_rmse_tst, std_rmse_tst), fontsize=22, fontdict={'weight': 'bold'} )
            ax[3].set_xlabel('error(min)', fontsize=22, fontdict={'weight': 'bold'})
            ax[3].set_ylabel('')
            ax[3].set_yticks([])
        plt.show()