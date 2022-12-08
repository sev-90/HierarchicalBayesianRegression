import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import geopandas as gpd
import math
import pickle
from datetime import datetime

import pymc3 as pm
import arviz as az
import arviz.labels as azl
from theano import shared
import theano.tensor as tt

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

class HBR_pymc(object):
    def __init__(self, data_type=None, test=False, pred=False, direct=False, logdirect=True, logN=False):
        self.data_type = data_type
        self.test = test
        self.direct = direct
        self.logdirect = logdirect
        self.logN = logN
        print('****** loading data...')
        self.load_data()
        print('****** prepare train input...')
        self.prepare_model_tarining_input()
        print('****** prepare test input...')
        self.prepare_model_testing_input()
        
        if self.logN: 
            print('****** constructing log model...')
            self.construct_logmodel()
        elif self.logdirect:
            print('****** constructing log direct model...')
            self.construct_direct_logmodel()
        elif self.direct:
            print('****** constructing direct model...')
            self.construct_direct_model()
        else:
            print('****** constructing model...')
            self.construct_model()
        ####
        if test:
            self.test_model()
        elif pred:
            self.load_model()
            self.load_test()
            self.plot_predictions()
            self.predictions = {}
            self.save_predictions()
        else:
            self.run_advi()
            # self.test_model()
            # self.load_model(model_name = 'model_1_itrace_2022-10-19 15:00:38_logn.nc') #'model_1_itrace_2022-06-07 11:52:51_logn.nc'
            # self.load_model(model_name = 'model_1_itrace_2022-06-08 14:58:18_logn.nc')
            # self.load_model(model_name = 'model_1_itrace_2022-10-19 16:37:39_logdirect.nc')
            # self.test_model()
            self.test_model_direct()
            # self.load_test(model_name = 'model1_MN_ppc_test2022-06-07 11:59:23_logn.pickle')
            # self.plot_predictions()
            self.save_predictions()

    
    def load_data(self):
        data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/data/modeling_data/{}/'.format(self.data_type)
        self.train_df = pd.read_csv(data_path + 'train_wpreds.txt') #'train_wpreds.txt'  train_wpreds_1

        # print(self.train_df['Orgn_Drgn'].unique())
        # exit()
        self.test_df= pd.read_csv(data_path + 'test_wpreds.txt')
        with open(data_path + 'scale_parameters.pickle', 'rb') as handle:
            scale_parameters = pickle.load(handle)
        self.variables_means = scale_parameters['means']
        self.variables_stds = scale_parameters['stds']
        # print(self.variables_means)
        # print(self.variables_stds)
        # exit()
    

    def prepare_model_tarining_input(self):  
        print('******* scaling inputs...')
        def scale_variable(variable, x):
            scaled_x = (x[variable] - self.variables_means[variable])/self.variables_stds[variable]
            return scaled_x
        self.path_avgBtwness = self.train_df[['path_avg_bwness','Orgn_Drgn']].apply(lambda x: scale_variable('path_avg_bwness', x), axis=1).values
        self.path_avgDegree = self.train_df[['path_avg_degree','Orgn_Drgn']].apply(lambda x: scale_variable('path_avg_degree', x), axis=1).values

        self.miles = self.train_df[['shortPath_mile','Orgn_Drgn']].apply(lambda x: scale_variable('shortPath_mile', x), axis=1).values
        self.unitDistance_TravelTimes = self.train_df['unitDist_TT_hpm'].values.T   
        self.travel_time = self.train_df['travel_time'].values.T/60. 
        self.regionPair_idx = self.train_df['Orgn_Drgn_code'].values.T
        self.hour_idx = self.train_df['hour'].values.T
        self.highway_idx = self.train_df['highway'].values.T
        self.weather_idx = self.train_df['adverse_weather'].values.T
    
    def prepare_model_testing_input(self):  
        print('******* scaling inputs...')
        def scale_variable(variable, x):
            scaled_x = (x[variable] - self.variables_means[variable])/self.variables_stds[variable]
            return scaled_x
        self.path_avgBtwness_tst = self.test_df[['path_avg_bwness','Orgn_Drgn']].apply(lambda x: scale_variable('path_avg_bwness', x), axis=1).values
        self.path_avgDegree_tst = self.test_df[['path_avg_degree','Orgn_Drgn']].apply(lambda x: scale_variable('path_avg_degree', x), axis=1).values
        self.miles_tst = self.test_df[['shortPath_mile','Orgn_Drgn']].apply(lambda x: scale_variable('shortPath_mile', x), axis=1).values

        self.unitDistance_TravelTimes_tst = self.test_df['unitDist_TT_hpm'].values.T    
        self.travel_time_tst = self.test_df['travel_time'].values.T/60. 
        self.regionPair_idx_tst = self.test_df['Orgn_Drgn_code'].values.T
        self.hour_idx_tst = self.test_df['hour'].values.T
        self.highway_idx_tst = self.test_df['highway'].values.T
        self.weather_idx_tst = self.test_df['adverse_weather'].values.T

    def construct_direct_logmodel(self):
        
        self.RANDOM_SEED = 3407
        np.random.seed(self.RANDOM_SEED)
        pm.set_tt_rng(self.RANDOM_SEED)
        # n_regionsPair = len(total_groups) #len(data['Oatom_Datom_code'].unique())
        total_groups = sorted(['South_South', 'North_Central', 'North_North', 'Central_Central',
                        'Central_North', 'Central_South', 'South_Central'])
        coords = {'region_pairs': total_groups,'hours': np.arange(24), 'obs_id': np.arange(len(self.train_df)), 'intercept':['intercept'],'highway':['highway'],
                        'Street':['Street mile'], 'bwness':['bwness centrality'], 'degree':['degree centrality'], 'adverse weather':['adverse weather'],
                        'miles':['miles']}  #,
        with pm.Model(coords = coords) as self.TT_model:

            self.Model_path_avgBtwness = pm.Data('Model_path_avgBtwness', self.path_avgBtwness, dims = "obs_id" )
            self.Model_path_avgDegree = pm.Data('Model_path_avgDegree', self.path_avgDegree, dims = "obs_id" )
            self.Model_miles = pm.Data('Model_miles', self.miles, dims = "obs_id" )
            self.Model_regionPair_idx = pm.Data('Model_regionPair_idx', self.regionPair_idx, dims = "obs_id" )
            self.Model_hour_idx = pm.Data('Model_hour_idx', self.hour_idx, dims = "obs_id" )
            self.Model_highway_idx = pm.Data('Model_highway_idx', self.highway_idx, dims="obs_id")
            self.Model_weather_idx = pm.Data('Model_weather_idx', self.weather_idx, dims="obs_id")


            b0 = pm.Normal('b0', mu = -2., sigma = 1.) 
            sigma_b0 = pm.HalfNormal('sigma_b0', 1.) ### intercept

            b1 = pm.Normal('b1', mu = -2., sigma = 1.) ## street slope
            sigma_b1 = pm.HalfNormal('sigma_b1', 1.) 

            b2 = pm.Normal('b2', mu = -2., sigma = 1.) ## highway slope
            sigma_b2 = pm.HalfNormal('sigma_b2', 1.) 

            b3 = pm.Normal('b3', mu = -2., sigma = 1.) ## btwness slope
            sigma_b3 = pm.HalfNormal('sigma_b3', 1.) 

            b4 = pm.Normal('b4', mu = -2., sigma = 1.) ## adverse weather
            sigma_b4 = pm.HalfNormal('sigma_b4', 1.)

            b5 = pm.Normal('b5', mu = -2., sigma = 1.) ## adverse weather
            sigma_b5 = pm.HalfNormal('sigma_b5', 1.)


            zb0_region = pm.Normal('zb0_region', mu = 0, sigma = 1, dims = ('hours','region_pairs'))
            b0_ = b0 + zb0_region[self.Model_hour_idx, self.Model_regionPair_idx] * sigma_b0

            zb1_region = pm.Normal('zb1_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b1_ = b1 + zb1_region[self.Model_regionPair_idx] * sigma_b1

            zb2_region = pm.Normal('zb2_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b2_ = b2 + zb2_region[self.Model_regionPair_idx] * sigma_b2

            zb3_region = pm.Normal('zb3_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b3_ = b3 + zb3_region[self.Model_regionPair_idx] * sigma_b3

            zb4_region = pm.Normal('zb4_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b4_ = b4 + zb4_region[self.Model_regionPair_idx] * sigma_b4

            zb5_region = pm.Normal('zb5_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b5_ = b5 + zb5_region[self.Model_regionPair_idx] * sigma_b5


            b0_region = pm.Deterministic('b0_region', b0 + zb0_region * sigma_b0, dims = ('hours','region_pairs'))  
            b1_region = pm.Deterministic('b1_region', b1 + zb1_region * sigma_b1, dims = 'region_pairs') 
            b2_region = pm.Deterministic('b2_region', b2 + zb2_region * sigma_b2, dims = 'region_pairs') 
            b3_region = pm.Deterministic('b3_region', b3 + zb3_region * sigma_b3, dims = 'region_pairs') 
            b4_region = pm.Deterministic('b4_region', b4 + zb4_region * sigma_b4, dims = 'region_pairs') 
            b5_region = pm.Deterministic('b5_region', b5 + zb5_region * sigma_b5, dims = 'region_pairs')

            
            Beta0 = pm.Normal('Beta0', mu = -2., sigma = 1., dims = 'intercept')
            Beta1 = pm.Normal('Beta1', mu = -2., sigma = 1., dims = 'highway' )
            Beta2 = pm.Normal('Beta2', mu = -2., sigma = 1., dims = 'bwness' )
            Beta3 = pm.Normal('Beta3', mu = -2., sigma = 1., dims = 'degree' )
            Beta4 = pm.Normal('Beta4', mu = -2., sigma = 1., dims = 'adverse weather' )
            Beta5 = pm.Normal('Beta5', mu = -2., sigma = 1., dims = 'miles' )

            theta = Beta0 + Beta1 * self.Model_highway_idx +  Beta2 * self.Model_path_avgBtwness  + Beta3 * self.Model_path_avgDegree + Beta4 * self.Model_weather_idx + Beta5 * self.Model_miles + \
                    b0_  +    b1_ * self.Model_highway_idx + b2_ * self.Model_path_avgBtwness  + b3_ * self.Model_path_avgDegree + b4_* self.Model_weather_idx + b5_* self.Model_miles #+ \
                    # bh0 #+ bh1 * Model_highway_idx + bh2 * Model_path_avgBtwness  + bh3 * Model_path_avgDegree
            sigma = pm.HalfNormal('sigma', 1., dims = 'region_pairs') #, dims = 'region_pairs'
            # g = pm.Normal('g', mu = 0, sigma = 1, shape=2)
            # sigma = g[0] + g[1] * theta
            
            # nu = pm.Gamma('nu', 2, 0.1) #nu=nu,
            y = pm.Lognormal('y', mu = theta, sigma = sigma[self.Model_regionPair_idx], observed = self.travel_time, dims= "obs_id" ) #sigma[Model_regionPair_idx] nu=4, 
            
            # y = pm.Deterministic('y', y_hat * Model_miles * 3600, dims= "obs_id")
            # sigma = pm.Deterministic('sigma', sigma_hat[Model_regionPair_idx] * np.sqrt( Model_miles) * 3600, dims= "obs_id") 

            self.prior_checks = pm.sample_prior_predictive(random_seed=self.RANDOM_SEED)
            # trace = pm.sample(random_seed = RANDOM_SEED, chains=2,  
            #                   draws=6000, tune=1000, return_inferencedata=True, target_accept=0.95)     
    def construct_direct_model(self):
        
        self.RANDOM_SEED = 3407
        np.random.seed(self.RANDOM_SEED)
        pm.set_tt_rng(self.RANDOM_SEED)
        # n_regionsPair = len(total_groups) #len(data['Oatom_Datom_code'].unique())
        total_groups = sorted(['South_South', 'North_Central', 'North_North', 'Central_Central',
                        'Central_North', 'Central_South', 'South_Central'])
        coords = {'region_pairs': total_groups,'hours': np.arange(24), 'obs_id': np.arange(len(self.train_df)), 'intercept':['intercept'],'highway':['highway'],
                        'Street':['Street mile'], 'bwness':['bwness centrality'], 'degree':['degree centrality'], 'adverse weather':['adverse weather'],
                        'miles':['miles']}  #,
        with pm.Model(coords = coords) as self.TT_model:

            self.Model_path_avgBtwness = pm.Data('Model_path_avgBtwness', self.path_avgBtwness, dims = "obs_id" )
            self.Model_path_avgDegree = pm.Data('Model_path_avgDegree', self.path_avgDegree, dims = "obs_id" )
            self.Model_miles = pm.Data('Model_miles', self.miles, dims = "obs_id" )
            self.Model_regionPair_idx = pm.Data('Model_regionPair_idx', self.regionPair_idx, dims = "obs_id" )
            self.Model_hour_idx = pm.Data('Model_hour_idx', self.hour_idx, dims = "obs_id" )
            self.Model_highway_idx = pm.Data('Model_highway_idx', self.highway_idx, dims="obs_id")
            self.Model_weather_idx = pm.Data('Model_weather_idx', self.weather_idx, dims="obs_id")


            b0 = pm.Normal('b0', mu = 0., sigma = 1.) 
            sigma_b0 = pm.HalfNormal('sigma_b0', 1.) ### intercept

            b1 = pm.Normal('b1', mu = 0., sigma = 1.) ## street slope
            sigma_b1 = pm.HalfNormal('sigma_b1', 1.) 

            b2 = pm.Normal('b2', mu = 0., sigma = 1.) ## highway slope
            sigma_b2 = pm.HalfNormal('sigma_b2', 1.) 

            b3 = pm.Normal('b3', mu = 0., sigma = 1.) ## btwness slope
            sigma_b3 = pm.HalfNormal('sigma_b3', 1.) 

            b4 = pm.Normal('b4', mu = 0., sigma = 1.) ## adverse weather
            sigma_b4 = pm.HalfNormal('sigma_b4', 1.)

            b5 = pm.Normal('b5', mu = 0., sigma = 1.) ## adverse weather
            sigma_b5 = pm.HalfNormal('sigma_b5', 1.)


            zb0_region = pm.Normal('zb0_region', mu = 0, sigma = 1, dims = ('hours','region_pairs'))
            b0_ = b0 + zb0_region[self.Model_hour_idx, self.Model_regionPair_idx] * sigma_b0

            zb1_region = pm.Normal('zb1_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b1_ = b1 + zb1_region[self.Model_regionPair_idx] * sigma_b1

            zb2_region = pm.Normal('zb2_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b2_ = b2 + zb2_region[self.Model_regionPair_idx] * sigma_b2

            zb3_region = pm.Normal('zb3_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b3_ = b3 + zb3_region[self.Model_regionPair_idx] * sigma_b3

            zb4_region = pm.Normal('zb4_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b4_ = b4 + zb4_region[self.Model_regionPair_idx] * sigma_b4

            zb5_region = pm.Normal('zb5_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b5_ = b5 + zb5_region[self.Model_regionPair_idx] * sigma_b5


            b0_region = pm.Deterministic('b0_region', b0 + zb0_region * sigma_b0, dims = ('hours','region_pairs'))  
            b1_region = pm.Deterministic('b1_region', b1 + zb1_region * sigma_b1, dims = 'region_pairs') 
            b2_region = pm.Deterministic('b2_region', b2 + zb2_region * sigma_b2, dims = 'region_pairs') 
            b3_region = pm.Deterministic('b3_region', b3 + zb3_region * sigma_b3, dims = 'region_pairs') 
            b4_region = pm.Deterministic('b4_region', b4 + zb4_region * sigma_b4, dims = 'region_pairs') 
            b5_region = pm.Deterministic('b5_region', b5 + zb5_region * sigma_b5, dims = 'region_pairs')

            
            Beta0 = pm.Normal('Beta0', mu = 5., sigma = 1., dims = 'intercept')
            Beta1 = pm.Normal('Beta1', mu = 0., sigma = 1., dims = 'highway' )
            Beta2 = pm.Normal('Beta2', mu = 0., sigma = 1., dims = 'bwness' )
            Beta3 = pm.Normal('Beta3', mu = 0., sigma = 1., dims = 'degree' )
            Beta4 = pm.Normal('Beta4', mu = 0., sigma = 1., dims = 'adverse weather' )
            Beta5 = pm.Normal('Beta5', mu = 0., sigma = 1., dims = 'miles' )

            theta = Beta0 + Beta1 * self.Model_highway_idx +  Beta2 * self.Model_path_avgBtwness  + Beta3 * self.Model_path_avgDegree + Beta4 * self.Model_weather_idx + Beta5 * self.Model_miles + \
                    b0_  +    b1_ * self.Model_highway_idx + b2_ * self.Model_path_avgBtwness  + b3_ * self.Model_path_avgDegree + b4_* self.Model_weather_idx + b5_* self.Model_miles #+ \
                    # bh0 #+ bh1 * Model_highway_idx + bh2 * Model_path_avgBtwness  + bh3 * Model_path_avgDegree
            sigma = pm.HalfNormal('sigma', 1, dims = 'region_pairs') #, dims = 'region_pairs'
            # g = pm.Normal('g', mu = 0, sigma = 1, shape=2)
            # sigma = g[0] + g[1] * theta
            
            # nu = pm.Gamma('nu', 2, 0.1) #nu=nu,
            y = pm.Normal('y', mu = theta, sigma = sigma[self.Model_regionPair_idx], observed = self.travel_time, dims= "obs_id" ) #sigma[Model_regionPair_idx] nu=4, 
            
            # y = pm.Deterministic('y', y_hat * Model_miles * 3600, dims= "obs_id")
            # sigma = pm.Deterministic('sigma', sigma_hat[Model_regionPair_idx] * np.sqrt( Model_miles) * 3600, dims= "obs_id") 

            self.prior_checks = pm.sample_prior_predictive(random_seed=self.RANDOM_SEED)
            # trace = pm.sample(random_seed = RANDOM_SEED, chains=2,  
            #                   draws=6000, tune=1000, return_inferencedata=True, target_accept=0.95) 
    def construct_logmodel(self):
        
        self.RANDOM_SEED = 3407
        np.random.seed(self.RANDOM_SEED)
        pm.set_tt_rng(self.RANDOM_SEED)
        # n_regionsPair = len(total_groups) #len(data['Oatom_Datom_code'].unique())
        total_groups = sorted(['South_South', 'North_Central', 'North_North', 'Central_Central',
                        'Central_North', 'Central_South', 'South_Central'])
        coords = {'region_pairs': total_groups,'hours': np.arange(24), 'obs_id': np.arange(len(self.train_df)), 'intercept':['intercept'],'highway':['highway'],
                        'Street':['Street mile'], 'bwness':['bwness centrality'], 'degree':['degree centrality'], 'adverse weather':['adverse weather']}  #,
        with pm.Model(coords = coords) as self.TT_model:

            self.Model_path_avgBtwness = pm.Data('Model_path_avgBtwness', self.path_avgBtwness, dims = "obs_id" )
            self.Model_path_avgDegree = pm.Data('Model_path_avgDegree', self.path_avgDegree, dims = "obs_id" )

            self.Model_regionPair_idx = pm.Data('Model_regionPair_idx', self.regionPair_idx, dims = "obs_id" )
            self.Model_hour_idx = pm.Data('Model_hour_idx', self.hour_idx, dims = "obs_id" )
            self.Model_highway_idx = pm.Data('Model_highway_idx', self.highway_idx, dims="obs_id")
            self.Model_weather_idx = pm.Data('Model_weather_idx', self.weather_idx, dims="obs_id")


            b0 = pm.Normal('b0', mu = -2., sigma = 1.)  #mu = -2.
            sigma_b0 = pm.HalfNormal('sigma_b0', 1) ### intercept

            b1 = pm.Normal('b1', mu = -2., sigma = 1.) ## street slope  #mu = -2.
            sigma_b1 = pm.HalfNormal('sigma_b1', 1) 

            b2 = pm.Normal('b2', mu = -2., sigma = 1.) ## highway slope #mu = -2.
            sigma_b2 = pm.HalfNormal('sigma_b2', 1) 

            b3 = pm.Normal('b3', mu = -2., sigma = 1.) ## btwness slope #mu = -2.
            sigma_b3 = pm.HalfNormal('sigma_b3', 1) 

            b4 = pm.Normal('b4', mu = -2., sigma = 1.) ## adverse weather #mu = -2.
            sigma_b4 = pm.HalfNormal('sigma_b4', 1)


            zb0_region = pm.Normal('zb0_region', mu = 0, sigma = 1, dims = ('hours','region_pairs'))
            b0_ = b0 + zb0_region[self.Model_hour_idx, self.Model_regionPair_idx] * sigma_b0

            zb1_region = pm.Normal('zb1_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b1_ = b1 + zb1_region[self.Model_regionPair_idx] * sigma_b1

            zb2_region = pm.Normal('zb2_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b2_ = b2 + zb2_region[self.Model_regionPair_idx] * sigma_b2

            zb3_region = pm.Normal('zb3_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b3_ = b3 + zb3_region[self.Model_regionPair_idx] * sigma_b3

            zb4_region = pm.Normal('zb4_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b4_ = b4 + zb4_region[self.Model_regionPair_idx] * sigma_b4


            b0_region = pm.Deterministic('b0_region', b0 + zb0_region * sigma_b0, dims = ('hours','region_pairs'))  
            b1_region = pm.Deterministic('b1_region', b1 + zb1_region * sigma_b1, dims = 'region_pairs') 
            b2_region = pm.Deterministic('b2_region', b2 + zb2_region * sigma_b2, dims = 'region_pairs') 
            b3_region = pm.Deterministic('b3_region', b3 + zb3_region * sigma_b3, dims = 'region_pairs') 
            b4_region = pm.Deterministic('b4_region', b4 + zb4_region * sigma_b4, dims = 'region_pairs') 

            
            Beta0 = pm.Normal('Beta0', mu = -2., sigma = 1., dims = 'intercept') #mu = -2.
            Beta1 = pm.Normal('Beta1', mu = -2., sigma = 1., dims = 'highway' ) #mu = -2.
            Beta2 = pm.Normal('Beta2', mu = -2., sigma = 1., dims = 'bwness' ) #mu = -2.
            Beta3 = pm.Normal('Beta3', mu = -2., sigma = 1., dims = 'degree' ) #mu = -2.
            Beta4 = pm.Normal('Beta4', mu = -2., sigma = 1., dims = 'adverse weather' ) #mu = -2.


            theta = Beta0 + Beta1 * self.Model_highway_idx +  Beta2 * self.Model_path_avgBtwness  + Beta3 * self.Model_path_avgDegree + Beta4 * self.Model_weather_idx + \
                    b0_  +    b1_ * self.Model_highway_idx + b2_ * self.Model_path_avgBtwness  + b3_ * self.Model_path_avgDegree + b4_* self.Model_weather_idx  #+ \
                    # bh0 #+ bh1 * Model_highway_idx + bh2 * Model_path_avgBtwness  + bh3 * Model_path_avgDegree
            sigma = pm.HalfNormal('sigma', 1, dims = 'region_pairs') #, dims = 'region_pairs'
            # g = pm.Normal('g', mu = 0, sigma = 1, shape=2)
            # sigma = g[0] + g[1] * theta
            
            # nu = pm.Gamma('nu', 2, 0.1) #nu=nu,
            y = pm.Lognormal('y', mu = theta, sigma = sigma[self.Model_regionPair_idx], observed = self.unitDistance_TravelTimes, dims= "obs_id" ) #sigma[Model_regionPair_idx] nu=4, 
            
            # y = pm.Deterministic('y', y_hat * Model_miles * 3600, dims= "obs_id")
            # sigma = pm.Deterministic('sigma', sigma_hat[Model_regionPair_idx] * np.sqrt( Model_miles) * 3600, dims= "obs_id") 

            self.prior_checks = pm.sample_prior_predictive(random_seed=self.RANDOM_SEED)
            # trace = pm.sample(random_seed = RANDOM_SEED, chains=2,  
            #                   draws=6000, tune=1000, return_inferencedata=True, target_accept=0.95)  
            # print("plot graphical model")
            # pm.model_to_graphviz(self.TT_model)           
    def construct_model(self):
        
        self.RANDOM_SEED = 3407
        np.random.seed(self.RANDOM_SEED)
        pm.set_tt_rng(self.RANDOM_SEED)
        # n_regionsPair = len(total_groups) #len(data['Oatom_Datom_code'].unique())
        total_groups = sorted(['South_South', 'North_Central', 'North_North', 'Central_Central',
                        'Central_North', 'Central_South', 'South_Central'])
        coords = {'region_pairs': total_groups,'hours': np.arange(24), 'obs_id': np.arange(len(self.train_df)), 'intercept':['intercept'],'highway':['highway'],
                        'Street':['Street mile'], 'bwness':['bwness centrality'], 'degree':['degree centrality'], 'adverse weather':['adverse weather']}  #,
        with pm.Model(coords = coords) as self.TT_model:

            self.Model_path_avgBtwness = pm.Data('Model_path_avgBtwness', self.path_avgBtwness, dims = "obs_id" )
            self.Model_path_avgDegree = pm.Data('Model_path_avgDegree', self.path_avgDegree, dims = "obs_id" )

            self.Model_regionPair_idx = pm.Data('Model_regionPair_idx', self.regionPair_idx, dims = "obs_id" )
            self.Model_hour_idx = pm.Data('Model_hour_idx', self.hour_idx, dims = "obs_id" )
            self.Model_highway_idx = pm.Data('Model_highway_idx', self.highway_idx, dims="obs_id")
            self.Model_weather_idx = pm.Data('Model_weather_idx', self.weather_idx, dims="obs_id")


            b0 = pm.Normal('b0', mu = 0., sigma = 1.) 
            sigma_b0 = pm.HalfNormal('sigma_b0', 0.5) ### intercept

            b1 = pm.Normal('b1', mu = 0., sigma = 1.) ## street slope
            sigma_b1 = pm.HalfNormal('sigma_b1', 0.5) 

            b2 = pm.Normal('b2', mu = 0., sigma = 1.) ## highway slope
            sigma_b2 = pm.HalfNormal('sigma_b2', 0.5) 

            b3 = pm.Normal('b3', mu = 0., sigma = 1.) ## btwness slope
            sigma_b3 = pm.HalfNormal('sigma_b3', 0.5) 

            b4 = pm.Normal('b4', mu = 0., sigma = 1.) ## adverse weather
            sigma_b4 = pm.HalfNormal('sigma_b4', 0.5)


            zb0_region = pm.Normal('zb0_region', mu = 0, sigma = 1, dims = ('hours','region_pairs'))
            b0_ = b0 + zb0_region[self.Model_hour_idx, self.Model_regionPair_idx] * sigma_b0

            zb1_region = pm.Normal('zb1_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b1_ = b1 + zb1_region[self.Model_regionPair_idx] * sigma_b1

            zb2_region = pm.Normal('zb2_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b2_ = b2 + zb2_region[self.Model_regionPair_idx] * sigma_b2

            zb3_region = pm.Normal('zb3_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b3_ = b3 + zb3_region[self.Model_regionPair_idx] * sigma_b3

            zb4_region = pm.Normal('zb4_region', mu = 0, sigma = 1, dims = 'region_pairs' )
            b4_ = b4 + zb4_region[self.Model_regionPair_idx] * sigma_b4


            b0_region = pm.Deterministic('b0_region', b0 + zb0_region * sigma_b0, dims = ('hours','region_pairs'))  
            b1_region = pm.Deterministic('b1_region', b1 + zb1_region * sigma_b1, dims = 'region_pairs') 
            b2_region = pm.Deterministic('b2_region', b2 + zb2_region * sigma_b2, dims = 'region_pairs') 
            b3_region = pm.Deterministic('b3_region', b3 + zb3_region * sigma_b3, dims = 'region_pairs') 
            b4_region = pm.Deterministic('b4_region', b4 + zb4_region * sigma_b4, dims = 'region_pairs') 

            
            Beta0 = pm.Normal('Beta0', mu = 0., sigma = 1., dims = 'intercept')
            Beta1 = pm.Normal('Beta1', mu = 0., sigma = 1., dims = 'highway' )
            Beta2 = pm.Normal('Beta2', mu = 0., sigma = 1., dims = 'bwness' )
            Beta3 = pm.Normal('Beta3', mu = 0., sigma = 1., dims = 'degree' )
            Beta4 = pm.Normal('Beta4', mu = 0., sigma = 1., dims = 'adverse weather' )

            theta = Beta0 + Beta1 * self.Model_highway_idx +  Beta2 * self.Model_path_avgBtwness  + Beta3 * self.Model_path_avgDegree + Beta4 * self.Model_weather_idx + \
                    b0_  +    b1_ * self.Model_highway_idx + b2_ * self.Model_path_avgBtwness  + b3_ * self.Model_path_avgDegree + b4_* self.Model_weather_idx  #+ \
                    # bh0 #+ bh1 * Model_highway_idx + bh2 * Model_path_avgBtwness  + bh3 * Model_path_avgDegree
            sigma = pm.HalfNormal('sigma', 0.5, dims = 'region_pairs') #, dims = 'region_pairs'
            # g = pm.Normal('g', mu = 0, sigma = 1, shape=2)
            # sigma = g[0] + g[1] * theta
            
            # nu = pm.Gamma('nu', 2, 0.1) #nu=nu,
            y = pm.Normal('y', mu = theta, sigma = sigma[self.Model_regionPair_idx], observed = self.unitDistance_TravelTimes, dims= "obs_id" ) #sigma[Model_regionPair_idx] nu=4, 
            
            # y = pm.Deterministic('y', y_hat * Model_miles * 3600, dims= "obs_id")
            # sigma = pm.Deterministic('sigma', sigma_hat[Model_regionPair_idx] * np.sqrt( Model_miles) * 3600, dims= "obs_id") 

            self.prior_checks = pm.sample_prior_predictive(random_seed=self.RANDOM_SEED)
            # trace = pm.sample(random_seed = RANDOM_SEED, chains=2,  
            #                   draws=6000, tune=1000, return_inferencedata=True, target_accept=0.95) 
    
    def run_advi(self):
        print('****** posterior inference with advi...')
        with self.TT_model:
            self.inference_model = pm.ADVI()
            self.approx_model = pm.fit(n=40000, method = self.inference_model, start=20000, random_seed = self.RANDOM_SEED) #40000
        save_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/trained_models/{}/hbr/'.format(self.data_type)
        if self.direct:
            with open(save_path + 'model1_MN_{}_direct.pickle'.format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')), 'wb') as buff:
                pickle.dump({'approx':self.approx_model,'inference':self.inference_model}, buff)
        else:
            with open(save_path + 'model1_MN_{}.pickle'.format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')), 'wb') as buff:
                pickle.dump({'approx':self.approx_model,'inference':self.inference_model}, buff)
        # with open('good_model_MN.pickle', 'rb') as buff:
            #     model_ = pickle.load(buff)
        # # model_ = model_['model']
        # approx_ = model_['approx']
        # infer_ = model_['inference']
        plt.rcParams["font.size"] = 12
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelsize"] = 18
        plt.rcParams["axes.labelweight"] = "bold"
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(-self.inference_model.hist,'dimgray') #label="new ADVI",
        # plt.plot(approx_model.hist, label="old ADVI", alpha=0.3)
        # ax.legend()
        ax.set_ylabel("ELBO" )
        ax.set_xlabel("Iteration")
        plt.show()

        self.trace = self.approx_model.sample(draws=10000)
        if self.logdirect:
            pm.save_trace(self.trace, directory=save_path + "model_1_trace_{}_logdirect.nc".format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')), overwrite=True)
        elif self.direct:
            pm.save_trace(self.trace, directory=save_path + "model_1_trace_{}_direct.nc".format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')), overwrite=True)
        elif self.logN:
            pm.save_trace(self.trace, directory=save_path + "model_1_trace_{}_logn.nc".format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')), overwrite=True)
        else:
            pm.save_trace(self.trace, directory=save_path + "model_1_trace_{}.nc".format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')), overwrite=True)

        print('****** posterior predictive sampling...')
        with self.TT_model:
            self.ppc = pm.sample_posterior_predictive(self.trace, var_names=['y'], samples=1000, random_seed=6)
        self.trace = az.from_pymc3(self.trace, prior = self.prior_checks, posterior_predictive = self.ppc, model = self.TT_model)
        print('****** saving model...')

        if self.logN:
            az.to_netcdf(self.trace, save_path + "model_1_itrace_{}_logn.nc".format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')))
        elif self.logdirect:
            az.to_netcdf(self.trace, save_path + "model_1_itrace_{}_logdirect.nc".format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')))
        elif self.direct:
            az.to_netcdf(self.trace, save_path + "model_1_itrace_{}_direct.nc".format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')))
        else:
            az.to_netcdf(self.trace, save_path + "model_1_itrace_{}.nc".format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')))
        self.ppc = self.trace.posterior_predictive
    
    def load_model(self, model_name):
        save_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/trained_models/{}/hbr/'.format(self.data_type)
        self.trace = az.from_netcdf(save_path + model_name) #"model_1_itrace_2022-06-01 16:16:09.nc"
        self.ppc = self.trace.posterior_predictive
        
    def test_model_direct(self):
        # print('****** loading trained model...')
        # self.load_model()
        print('****** testing model...')
        with self.TT_model:
            self.Model_path_avgBtwness.set_value(self.path_avgBtwness_tst)
            self.Model_path_avgDegree.set_value(self.path_avgDegree_tst)
            self.Model_highway_idx.set_value(self.highway_idx_tst.astype('int32'))
            self.Model_weather_idx.set_value(self.weather_idx_tst.astype('int32'))
            self.Model_miles.set_value(self.miles_tst) 

            self.Model_regionPair_idx.set_value(self.regionPair_idx_tst.astype('int32'))
            self.Model_hour_idx.set_value(self.hour_idx_tst.astype('int32'))
            self.ppc_test = pm.sample_posterior_predictive(self.trace, var_names=['y'], samples=5000, random_seed=6)
        save_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/trained_models/{}/hbr/'.format(self.data_type)
        if self.logdirect:
            with open(save_path + 'model1_MN_ppc_test{}_logdirect.pickle'.format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')), 'wb') as buff:
                pickle.dump(self.ppc_test, buff)
        else:
            with open(save_path + 'model1_MN_ppc_test{}_direct.pickle'.format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')), 'wb') as buff:
                pickle.dump(self.ppc_test, buff)
    
    def test_model(self):
        print('****** loading trained model...')
        # self.load_model(model_name='model_1_itrace_2022-10-19 21:35:20_logn.nc') #'model_1_itrace_2022-07-12 12:42:09_logn.nc'
        print('****** testing model...')
        with self.TT_model:
            self.Model_path_avgBtwness.set_value(self.path_avgBtwness_tst)
            self.Model_path_avgDegree.set_value(self.path_avgDegree_tst)
            self.Model_highway_idx.set_value(self.highway_idx_tst.astype('int32'))
            self.Model_weather_idx.set_value(self.weather_idx_tst.astype('int32'))
            
            self.Model_regionPair_idx.set_value(self.regionPair_idx_tst.astype('int32'))
            self.Model_hour_idx.set_value(self.hour_idx_tst.astype('int32'))
            self.ppc_test = pm.sample_posterior_predictive(self.trace, var_names=['y'], samples=5000, random_seed=6)
        save_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/trained_models/{}/hbr/'.format(self.data_type)
        if self.logN:
            with open(save_path + 'model1_MN_ppc_test{}_logn.pickle'.format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')), 'wb') as buff:
                pickle.dump(self.ppc_test, buff)
        else:
            with open(save_path + 'model1_MN_ppc_test{}.pickle'.format(datetime.today().strftime('%Y-%m-%d %H:%M:%S')), 'wb') as buff:
                pickle.dump(self.ppc_test, buff)
    
    def load_test(self, model_name):
        save_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/trained_models/{}/hbr/'.format(self.data_type)
        with open(save_path + model_name, 'rb') as buff:
            self.ppc_test = pickle.load(buff) 
    
    def plot_predictions(self):
        nplot = 100
        if self.logdirect:
            fig, ax = plt.subplots(ncols = 2, figsize=(10,5))
            ax[0].plot(self.travel_time[:nplot], self.ppc['y'].mean(dim=('chain', 'draw'))[:nplot],'.' )
            ax[0].errorbar(x = self.travel_time[:nplot], y = self.ppc['y'].mean(dim=('chain', 'draw'))[:nplot], yerr= self.ppc['y'].std(dim=('chain', 'draw'))[:nplot], fmt='r.', alpha=0.4)
            ax[0].plot([0,60],[0,60])
            ax[0].set_xlabel('True')
            ax[0].set_ylabel('Pred')
            ax[1].plot(self.travel_time_tst[:nplot], self.ppc_test['y'].mean(axis=0)[:nplot],'.' )
            ax[1].errorbar(x = self.travel_time_tst[:nplot], y = self.ppc_test['y'].mean(axis=0)[:nplot], yerr= self.ppc_test['y'].std(axis=0)[:nplot], fmt='r.', alpha=0.4)
            ax[1].plot([0,60],[0,60])
            ax[1].set_xlabel('True')
            ax[1].set_ylabel('Pred')
            plt.show()
        elif self.direct:
            fig, ax = plt.subplots(ncols = 2, figsize=(10,5))
            ax[0].plot(self.travel_time[:nplot], self.ppc['y'].mean(dim=('chain', 'draw'))[:nplot],'.' )
            ax[0].errorbar(x = self.travel_time[:nplot], y = self.ppc['y'].mean(dim=('chain', 'draw'))[:nplot], yerr= self.ppc['y'].std(dim=('chain', 'draw'))[:nplot], fmt='r.', alpha=0.4)
            ax[0].plot([0,60],[0,60])
            ax[0].set_xlabel('True')
            ax[0].set_ylabel('Pred')
            ax[1].plot(self.travel_time_tst[:nplot], self.ppc_test['y'].mean(axis=0)[:nplot],'.' )
            ax[1].errorbar(x = self.travel_time_tst[:nplot], y = self.ppc_test['y'].mean(axis=0)[:nplot], yerr= self.ppc_test['y'].std(axis=0)[:nplot], fmt='r.', alpha=0.4)
            ax[1].plot([0,60],[0,60])
            ax[1].set_xlabel('True')
            ax[1].set_ylabel('Pred')
            plt.show()
        else:
            fig, ax = plt.subplots(ncols = 2, figsize=(10,5))
            ax[0].plot(self.unitDistance_TravelTimes[:nplot], self.ppc['y'].mean(dim=('chain', 'draw'))[:nplot],'.' )
            ax[0].errorbar(x = self.unitDistance_TravelTimes[:nplot], y = self.ppc['y'].mean(dim=('chain', 'draw'))[:nplot], yerr= self.ppc['y'].std(dim=('chain', 'draw'))[:nplot], fmt='r.', alpha=0.4)
            ax[0].plot([0,0.25],[0,0.25])
            ax[0].set_xlabel('True')
            ax[0].set_ylabel('Pred')
            ax[1].plot(self.unitDistance_TravelTimes_tst[:nplot], self.ppc_test['y'].mean(axis=0)[:nplot],'.' )
            ax[1].errorbar(x = self.unitDistance_TravelTimes_tst[:nplot], y = self.ppc_test['y'].mean(axis=0)[:nplot], yerr= self.ppc_test['y'].std(axis=0)[:nplot], fmt='r.', alpha=0.4)
            ax[1].plot([0,0.25],[0,0.25])
            ax[1].set_xlabel('True')
            ax[1].set_ylabel('Pred')
            plt.show()
    
    def save_predictions(self):
        if self.logN:
            
            self.train_df['hbr_udtt_predMean_logn'] = self.ppc['y'].mean(dim=('chain', 'draw'))
            self.train_df['hbr_udtt_predStd_logn'] = self.ppc['y'].std(dim=('chain', 'draw'))

            self.test_df['hbr_udtt_predMean_logn'] = self.ppc_test['y'].mean(axis=0)
            self.test_df['hbr_udtt_predStd_logn'] = self.ppc_test['y'].std(axis=0)

            miles = self.train_df['shortPath_mile'].values
            self.train_df['hbr_tt_predMean_logn'] = self.train_df.hbr_udtt_predMean_logn * miles * 60
            self.train_df['hbr_tt_predStd_logn'] = self.train_df.hbr_udtt_predStd_logn * np.sqrt( miles) * 60


            miles = self.test_df['shortPath_mile'].values
            self.test_df['hbr_tt_predMean_logn'] = self.test_df.hbr_udtt_predMean_logn * miles * 60
            self.test_df['hbr_tt_predStd_logn'] = self.test_df.hbr_udtt_predStd_logn * np.sqrt( miles) * 60
            
        elif self.logdirect:
            self.train_df['hbr_tt_predMean_logdirect'] = self.ppc['y'].mean(dim=('chain', 'draw'))
            self.train_df['hbr_tt_predStd_logdirect'] = self.ppc['y'].std(dim=('chain', 'draw'))
            self.test_df['hbr_tt_predMean_logdirect'] = self.ppc_test['y'].mean(axis=0)
            self.test_df['hbr_tt_predStd_logdirect'] = self.ppc_test['y'].std(axis=0)

        elif self.direct:
            self.train_df['hbr_tt_predMean_direct'] = self.ppc['y'].mean(dim=('chain', 'draw'))
            self.train_df['hbr_tt_predStd_direct'] = self.ppc['y'].std(dim=('chain', 'draw'))

            self.test_df['hbr_tt_predMean_direct'] = self.ppc_test['y'].mean(axis=0)
            self.test_df['hbr_tt_predStd_direct'] = self.ppc_test['y'].std(axis=0)
        else:
        
            self.train_df['hbr_udtt_predMean'] = self.ppc['y'].mean(dim=('chain', 'draw'))
            self.train_df['hbr_udtt_predStd'] = self.ppc['y'].std(dim=('chain', 'draw'))

            self.test_df['hbr_udtt_predMean'] = self.ppc_test['y'].mean(axis=0)
            self.test_df['hbr_udtt_predStd'] = self.ppc_test['y'].std(axis=0)

            miles = self.train_df['shortPath_mile'].values
            self.train_df['hbr_tt_predMean'] = self.train_df.hbr_udtt_predMean * miles * 60
            self.train_df['hbr_tt_predStd'] = self.train_df.hbr_udtt_predStd * np.sqrt( miles) * 60


            miles = self.test_df['shortPath_mile'].values
            self.test_df['hbr_tt_predMean'] = self.test_df.hbr_udtt_predMean * miles * 60
            self.test_df['hbr_tt_predStd'] = self.test_df.hbr_udtt_predStd * np.sqrt( miles) * 60


        data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/data/modeling_data/{}/'.format(self.data_type)
        self.train_df.to_csv(data_path + 'train_wpreds.txt')
        self.test_df.to_csv(data_path + 'test_wpreds.txt')