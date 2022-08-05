import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import geopandas as gpd
import math
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

from datetime import datetime

from ast import literal_eval
from itertools import product

from shapely.geometry import Polygon, Point,LineString

from sklearn.model_selection import train_test_split
import networkx as nx
# from amb_dataProcessing import *


class data_processing(object):
    def __init__(self, data_type=None):
        self.data_type = data_type
        # shape file info:
        shapefile_path = '/home/sevin/Documents/Research/Data/Data for Columbia/Shapefiles/'    # path to parent data folder
        atoms_cent_shapename = 'Atom_Centroids_20190822.shp'
        atoms_pol_shapename = 'EMS_Atoms.shp'
        self.atoms_centroid_filename = shapefile_path + atoms_cent_shapename 
        self.atoms_pol_filename = shapefile_path + atoms_pol_shapename

        # raw data path:
        print('********** loading raw {} data...'.format(self.data_type))
        data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/data/raw_data/{}/data.txt'.format(self.data_type)
        self.data = pd.read_csv(data_path, index_col=0, dtype={'OriginNodes' : str, 'DestNodes' : str}).rename(columns={'From':'Oatom', 'To':'Datom'})
        ## 2016-01-01 00:00:17 2016-06-03 23:59:59 [6 2 5 3 4 1]
        # print(self.data.head(2))
        # print(pd.to_datetime(self.data['pickup_datetime']).min(), pd.to_datetime(self.data['pickup_datetime']).max(), pd.to_datetime(self.data['pickup_datetime']).dt.month.unique())
        # exit()
        self.raw_data_size = len(self.data)
        print('****** len raw data is {}'.format(self.raw_data_size))
        print('********** creating columns...')
        self.create_columns()
        print('********** fusing weather data...')
        self.merge_weather_data()
        print('********** calculate some stats...')
        self.calculate_stats()
        self.save_data()
    def concat_data(self):
        ### here u need to run networkx to extract shortest path and then append data as next lines
        if self.data_type == 'amb':
            dfs = []
            num_data = 15
            for i in range(num_data):
                data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/GP/Data/amb_data/'
                df = pd.read_csv(data_path + 'data_to_inc_MN_years[18to19]_seg[2-3]_complete_day123_wSPdist_{}.txt'.format(i))
                dfs.append(df)
                data = pd.concat(dfs)
            data.to_csv(data_path + 'data_to_inc_MN_years[18to19]_seg[2-3]_complete_day123_wSPdist.txt')
        if self.data_type == 'taxi':
            ### read and append data
            dfs = []
            num_data = 15
            for i in range(num_data):
                    data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/GP/Data/nyc-taxi-trip-duration/' + 'taxi_data_Tue_Thr_wSPdist_{}.txt'.format(i)
                    df = pd.read_csv(data_path)
                    dfs.append(df)
                    data = pd.concat(dfs)
            print(len(data))

            df_atoms, _ , region_groups = self.get_atoms_of_regions() # a dictionary 
            crs = {'init': 'epsg:4326'}
            gdf = gpd.GeoDataFrame(
                    data, crs=crs, geometry=gpd.points_from_xy(data.long_origin, data.lat_origin))
            gdf =  gdf.to_crs({'init': 'epsg:2263'})
            del gdf['index_right']
            new_gdf = gpd.sjoin(gdf, df_atoms[['ATOM','geometry']], op='within')


            new_gdf.rename(columns={'ATOM':'Oatom'}, inplace=True)
            del new_gdf['index_right']
            del new_gdf['geometry']

            gdf = gpd.GeoDataFrame(
                    new_gdf, crs=crs, geometry=gpd.points_from_xy(new_gdf.long_destination, new_gdf.lat_destination))
            gdf =  gdf.to_crs({'init': 'epsg:2263'})
            # del gdf['index_right']
            new_gdf = gpd.sjoin(gdf, df_atoms[['ATOM','geometry']], op='within')
            new_gdf.rename(columns={'ATOM':'Datom'}, inplace=True)

            new_gdf['pickup_datetime'] = pd.to_datetime(new_gdf['pickup_datetime'])
            new_gdf['hour'] = new_gdf['pickup_datetime'].dt.hour

            new_gdf.to_csv('/home/sevin/Desktop/projects/TravelTime_Prediction/GP/Data/nyc-taxi-trip-duration/' + 'taxi_data_Tue_Thr_wSPdist_processed.txt')
    
    def get_atoms_of_regions(self): # create dictionary that maps each atom to the region it belongs 
        atomsCenters_df = gpd.read_file(self.atoms_centroid_filename)
        self.atomsCenters_mn = atomsCenters_df[atomsCenters_df['AGENCY']=='MN'].reset_index(drop=True)
        self.atomsCenters_mn.rename(columns ={'geometry':'centroid'}, inplace=True) ## in dataframe format
        self.MN_atoms = list(self.atomsCenters_mn['ATOM'].unique())

        atoms_polygon_df = gpd.read_file(self.atoms_pol_filename) # atom polygons

        self.atoms_polygon_mn = atoms_polygon_df[atoms_polygon_df['ATOM'].isin(self.MN_atoms)].reset_index(drop=True)
        def change_names(x):
            return x.split(' ')[1]
        self.atoms_polygon_mn['DISP_FREQ'] = self.atoms_polygon_mn['DISP_FREQ'].apply(lambda x: change_names(x))
        # fig,ax = plt.subplots()
        # df_atoms.plot(facecolor='b', ax=ax)
        # d = df_atoms[df_atoms['ATOM'].isin(['WTLIB','WTELI','WARDB','RANDI','RANDH','076BC','GOVIA','RANDB','RANDK','WARDA',
        #                                                                         'RANDG','RANDJ','RANDF','RANDE','RANDD','RANDA','RANDC','076BC','025CE','025CB','025CF','040AE',
        #                                                                         '001BA','019DG','025CA','025CD','025CC','019DH','032FB','034IB','050IB',
        #                                                                         '078EA', '050GC', '050FA', '050EC', '103BD', '046MA', '114AB',
        #                                                                         '050ED', '052AC', '048BC', '040IA', '044TA', '050HA',
        #                                                                         '088JA', '090KB', '114ED', '040JK', '040DA', '077BB', '090JA',
        #                                                                         '044AA', '084FA', '084FD', '114FB', '050DA', '052AA', '078CA',
        #                                                                         '044FA', '083JB', '050EB', '103IB', '047BA', '044EB', '040JN',
        #                                                                         '052GA', '040FA', '050DB', '044AB', '044MB', '044PA', '040CA',
        #                                                                         '044DB', '106GB', '043HA', '075FF', '052CA', '114FC', '071JA',
        #                                                                         '084GA', '084CC', '110AA', '110BB', '048AB', '040GA', '044KC',
        #                                                                         '090FA', '084FC', '107BA', '084CA', '084DD', '079CA', '102DB',
        #                                                                         '050JB', '078BA', '044KA', '052BA', '050GA','046PA'])]
        # d.plot(facecolor='r', ax=ax)
        self.atoms_region = dict(zip(self.atoms_polygon_mn['ATOM'], self.atoms_polygon_mn['DISP_FREQ']))


    def clean(self):
        if 'wavg_speed' in self.data.columns:
            del self.data['wavg_speed']
        self.data.dropna(axis=0, how='any', inplace=True)
        print('*** droped {} rows due to NAs'.format(self.raw_data_size - len(self.data)))

        data_size = len(self.data)
        self.data['shortPath_NodesList'] = self.data['shortPath_NodesList'].apply(lambda x: literal_eval(x))  
        self.data = self.data[self.data.apply(lambda x: len(x['shortPath_NodesList'])!=1, axis=1)].reset_index(drop=True) ## removes no data
        print('*** droped {} rows due to path nodes with size 1'.format(data_size - len(self.data)))

        data_size = len(self.data)
        self.data = self.data[self.data['travel_time']>=1].reset_index(drop=True) ## removes 455 trips
        print('*** droped {} rows due to travel times less than 30 sec'.format(data_size - len(self.data)))

        data_size = len(self.data)
        self.data = self.data[self.data['unitDist_TT_hpm']>= 1/80] # will drop 138(348) data points/trips if we assume at max avg speed is 80 mph
        print('*** droped {} rows due to speeds greater than 80mph'.format(data_size - len(self.data)))

        data_size = len(self.data)
        self.data = self.data[self.data['unitDist_TT_hpm']<= 1/4] # will drop 4024(4013) data points/trips if we assume at min avg speed is 4 mph
        print('*** droped {} rows due to speeds smaller than 4mph'.format(data_size - len(self.data)))

        data_size = len(self.data)
        ### following will exclude 1334 trips belong to those atoms ###
        self.data = self.data[~self.data['Oatom'].isin(['WTLIB','WTELI','WARDB','RANDI','RANDH','076BC','GOVIA','RANDB','RANDK','WARDA',
                                                                                    'RANDG','RANDJ','RANDF','RANDE','RANDD','RANDA','RANDC','076BC','025CE','025CB','025CF','040AE',
                                                                                    '001BA','019DG','025CA','025CD','025CC','019DH','032FB','050IB',
                                                                                    '078EA', '050GC', '050FA', '050EC', '103BD', '046MA', '114AB',
                                                                                    '050ED', '052AC', '048BC', '040IA', '044TA', '050HA',
                                                                                    '088JA', '090KB', '114ED', '040JK', '040DA', '077BB', '090JA',
                                                                                    '044AA', '084FA', '084FD', '114FB', '050DA', '052AA', '078CA',
                                                                                    '044FA', '083JB', '050EB', '103IB', '047BA', '044EB', '040JN',
                                                                                    '052GA', '040FA', '050DB', '044AB', '044MB', '044PA', '040CA',
                                                                                    '044DB', '106GB', '043HA', '075FF', '052CA', '114FC', '071JA',
                                                                                    '084GA', '084CC', '110AA', '110BB', '048AB', '040GA', '044KC',
                                                                                    '090FA', '084FC', '107BA', '084CA', '084DD', '079CA', '102DB',
                                                                                    '050JB', '078BA', '044KA', '052BA', '050GA','046PA','005BA'])] 
        self.data = self.data[~self.data['Datom'].isin(['WTLIB','WTELI','WARDB','RANDI','RANDH','076BC','GOVIA','RANDB','RANDK','WARDA',
                                                                                    'RANDG','RANDJ','RANDF','RANDE','RANDD','RANDA','RANDC','076BC','025CE','025CB','025CF','040AE',
                                                                                    '001BA','019DG','025CA','025CD','025CC','019DH','032FB','050IB',
                                                                                    '078EA', '050GC', '050FA', '050EC', '103BD', '046MA', '114AB',
                                                                                    '050ED', '052AC', '048BC', '040IA', '044TA', '050HA',
                                                                                    '088JA', '090KB', '114ED', '040JK', '040DA', '077BB', '090JA',
                                                                                    '044AA', '084FA', '084FD', '114FB', '050DA', '052AA', '078CA',
                                                                                    '044FA', '083JB', '050EB', '103IB', '047BA', '044EB', '040JN',
                                                                                    '052GA', '040FA', '050DB', '044AB', '044MB', '044PA', '040CA',
                                                                                    '044DB', '106GB', '043HA', '075FF', '052CA', '114FC', '071JA',
                                                                                    '084GA', '084CC', '110AA', '110BB', '048AB', '040GA', '044KC',
                                                                                    '090FA', '084FC', '107BA', '084CA', '084DD', '079CA', '102DB',
                                                                                    '050JB', '078BA', '044KA', '052BA', '050GA','046PA','005BA'])] #'034IB',

        # self.data.dropna(axis=0, how='any', inplace=True) ## will exclude no data 
        self.data = self.data.reset_index(drop=True)
        print('*** droped {} rows due to corrupted atoms(out of range atoms'.format(data_size - len(self.data)))

    def create_indices(self):
        total_groups = self.data['Orgn_Drgn'].unique()
        print(" *** we have {} unique groups/atomPairs ****".format(len(total_groups)))
        total_groups = sorted(total_groups)
        self.indices_grps = {}
        for ind, obj in enumerate(total_groups):
            self.indices_grps.update({obj:ind})
        print(" *** we have {} unique index for each unique groups/regionPair ****".format(len(self.indices_grps)))

        # indices = dict(zip(ODs, range(ODs)))
        all_regions = list(set(np.append(self.data['Orgn'].unique(), self.data['Drgn'].unique())))
        regions = sorted(all_regions)
        self.indices_regions = {}
        for ind, obj in enumerate(regions):
            self.indices_regions.update({obj:ind})
        # indices = dict(zip(ODs, range(ODs)))
        print(" *** we have {} unique index for each unique region ****".format(len(self.indices_regions)))

    def load_graph(self):
        graph_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/data/graph_data/'
        with open(graph_path + 'network_graph_MN.pickle', 'rb') as handle:
            self.network_graph_ = pickle.load(handle)
        self.network_nodPos = self.network_graph_['nodPos']
        self.network_graph = self.network_graph_['graph']
        print('*** number of nodes in graph ', len(self.network_graph.nodes)) #number of nodes in graph  11587
        self.inDegreeCentr = nx.in_degree_centrality(self.network_graph)
        self.outDegreeCentr = nx.out_degree_centrality(self.network_graph)
        with open(graph_path + 'btwness_MN_k10000.pickle', 'rb') as handle:
            self.btwness = pickle.load(handle)

    def get_path_btness(self,x):
        sum_btness = 0
        cnt = 0
        try:
            for nd in x:
                if len(nd.split('_'))==1:
                    try:
                        cnt += 1
                        sum_btness += self.btwness[nd]
                    except:
                        pass
                else:
                    pass
            return sum_btness/cnt
        except:
            return None
    def get_path_degree(self, x):
        sum_degree = 0
        cnt = 0
        try:
            for nd in x:
                if len(nd.split('_'))==1:
                    try:
                        cnt += 1
                        sum_degree += self.outDegreeCentr[nd]
                        sum_degree += self.inDegreeCentr[nd]
                    except:
                        pass
                else:
                    pass
            return sum_degree/cnt * (len(self.network_graph.nodes)-1)
        except:
            return None

    def create_columns(self):
        self.data['unitDist_TT_hpm'] = self.data['travel_time'] / self.data['shortPath_mile']/3600
        print('****** fist cleaning data...')
        self.clean()
        self.data['Oatom_Datom'] = self.data['Oatom'] + '_' + self.data['Datom']

        self.get_atoms_of_regions()
        self.data['Orgn'] = self.data['Oatom'].apply(lambda x: self.atoms_region[x])
        self.data['Drgn'] = self.data['Datom'].apply(lambda x: self.atoms_region[x])
        self.data['Orgn_Drgn'] = self.data['Orgn'].astype(str) + '_' + self.data['Drgn'].astype(str)
        self.data = self.data[~self.data['Orgn_Drgn'].isin(['North_South','South_North'])].reset_index(drop=True)
        if 'start_time' in self.data.columns:
            self.data['start_time'] = pd.to_datetime(self.data['start_time'])
        elif 'pickup_datetime' in self.data.columns:
            self.data['start_time'] = pd.to_datetime(self.data['pickup_datetime'])
        else:
            pass
        if 'end_time' in self.data.columns:
            self.data['end_time'] = pd.to_datetime(self.data['end_time'])
        elif 'dropoff_datetime' in self.data.columns:
            self.data['end_time'] = pd.to_datetime(self.data['dropoff_datetime'])
        else:
            pass
        self.data['date'] = self.data['start_time'].dt.date
        self.data['minute'] = self.data['start_time'].dt.minute
        self.data['theta'] = self.data[['hour','minute']].apply(lambda x: (x.hour * 60 + x.minute) * 2 * np.pi /(24 * 60), axis=1)
        self.data['sin_theta'] = self.data['theta'].apply(lambda x: np.sin(x))
        self.data['cos_theta'] = self.data['theta'].apply(lambda x: np.cos(x))
        print('****** creating indices...')
        self.create_indices()
        self.data['Orgn_Drgn_code'] = self.data['Orgn_Drgn'].apply(lambda x: self.indices_grps[x])

        print('****** loading garph...')
        self.load_graph()
        self.data['path_avg_bwness'] = self.data['shortPath_NodesList'].apply(lambda x: self.get_path_btness(x))
        self.data['path_avg_degree'] = self.data['shortPath_NodesList'].apply(lambda x: self.get_path_degree(x))


    def load_weather_data(self):
        self.w_data = pd.read_csv('/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/data/raw_data/weather/2988065.csv')
        self.w_data = self.w_data[['DATE','HourlyPrecipitation','HourlyPresentWeatherType']]
        self.w_data['DATE'] = pd.to_datetime(self.w_data['DATE'])
        self.w_data['hour'] = self.w_data['DATE'].dt.hour
        self.w_data = self.w_data.drop_duplicates()

        def get_weatherType(x):
            # print(x)
            try:
                math.isnan(float(x))
                return 0
            except:
                return 1
        self.w_data['adverse_weather'] = self.w_data['HourlyPresentWeatherType'].apply(lambda x: get_weatherType(x))
        self.w_data['date'] = self.w_data['DATE'].dt.date
        # self.w_data['date_time'] = self.w_data['DATE'].dt.date pd.to_datetime(self.data['DATE'])
    
    def merge_weather_data(self):
        self.load_weather_data()
        # print(self.w_data['HourlyPresentWeatherType'].unique() )
        # exit()
        print('before',len(self.data))
        def fun_(x):
            # print(x)
            start = x['start_time']
            end = x['end_time']
            weat_subset = self.w_data[(self.w_data['DATE']>=start) & (self.w_data['DATE']<=end)]
            # print(weat_subset)
            # weat_subset = weat_subset[weat_subset['date_time']<=end]
            if len(weat_subset) == 0:
                return 0
            elif len(weat_subset) > 1:
                # assert len(weat_subset['adverse_weather'].unique()) ==1
                if 1 in weat_subset['adverse_weather'].unique():
                    return 1
                else:
                    return 0
            else:
                # print(weat_subset['adverse_weather'].values)
                return weat_subset['adverse_weather'].values.item()

        self.data['adverse_weather'] = self.data.apply(lambda x: fun_(x), axis=1)
        # print(self.data)
        # exit()
        
        # self.data = pd.merge(self.data, self.w_data[['hour','adverse_weather','date']],  how='left', left_on=['date','hour'], right_on = ['date','hour'])
        self.data.loc[self.data['adverse_weather'].isna(), 'adverse_weather'] = 0
        print('after',len(self.data))
        data_size = len(self.data)
        self.data.dropna(axis=0, how='any', inplace=True)
        print('*** droped {} rows due to speeds NAs'.format(data_size - len(self.data)))

    def calculate_stats(self):
        self.groups_obs_TT_means = dict(self.data.groupby('Orgn_Drgn')['travel_time'].mean())
        self.groups_obs_TT_stds = dict(self.data.groupby('Orgn_Drgn')['travel_time'].std())
        self.groups_obs_counts = dict(self.data.groupby('Orgn_Drgn')['Orgn_Drgn'].count())
        self.groups_obs_TT_means, self.groups_obs_TT_stds, self.groups_obs_counts

        self.groups_obs_normtime_means = dict(self.data.groupby('Orgn_Drgn')['unitDist_TT_hpm'].mean())
        self.groups_obs_normtime_stds = dict(self.data.groupby('Orgn_Drgn')['unitDist_TT_hpm'].std())
        self.groups_obs_normtime_means, self.groups_obs_normtime_stds

        self.groups_obs_bwness_means = dict(self.data.groupby('Orgn_Drgn')['path_avg_bwness'].mean())
        self.groups_obs_bwness_stds = dict(self.data.groupby('Orgn_Drgn')['path_avg_bwness'].std())

        self.groups_obs_degree_means = dict(self.data.groupby('Orgn_Drgn')['path_avg_degree'].mean())
        self.groups_obs_degree_stds = dict(self.data.groupby('Orgn_Drgn')['path_avg_degree'].std())

        self.groups_obs_strtMiles_means = dict(self.data.groupby('Orgn_Drgn')['street_miles'].mean())
        self.groups_obs_strtMiles_stds = dict(self.data.groupby('Orgn_Drgn')['street_miles'].std())

        self.groups_obs_highMiles_means = dict(self.data.groupby('Orgn_Drgn')['highway_miles'].mean())
        self.groups_obs_highMiles_stds = dict(self.data.groupby('Orgn_Drgn')['highway_miles'].std())

        self.groups_obs_mile_means = dict(self.data.groupby('Orgn_Drgn')['shortPath_mile'].mean())
        self.groups_obs_mile_stds = dict(self.data.groupby('Orgn_Drgn')['shortPath_mile'].std())

        self.variables = ['travel_time','highway_miles','street_miles','shortPath_mile', 'unitDist_TT_hpm','path_avg_degree','path_avg_bwness',
                'lat_origin','long_origin','lat_destination','long_destination',
                'x_origin',	'y_origin',	'x_destination', 'y_destination', 'sin_theta','cos_theta'] 
        self.variables_means = {}
        self.variables_stds = {}
        for var in self.variables:
            if var=='sin_theta':
                thetas = np.linspace(0.5, 24*60-0.5, 24*60) * 2 * np.pi / (24*60)
                mean = np.mean(np.sin(thetas))
                std = np.std(np.sin(thetas))
                self.variables_means.update({var:mean})
                self.variables_stds.update({var:std})
            elif var == 'cos_theta':
                thetas = np.linspace(0.5, 24*60-0.5, 24*60) * 2 * np.pi / (24*60)
                mean = np.mean(np.cos(thetas))
                std = np.std(np.cos(thetas))
                self.variables_means.update({var:mean})
                self.variables_stds.update({var:std})
            else:
                mean = np.mean(self.data[var].values)
                std = np.std(self.data[var].values)
                self.variables_means.update({var:mean})
                self.variables_stds.update({var:std})
        self.scale_parameters = {'means': self.variables_means, 'stds': self.variables_stds }
        print(self.data[['travel_time', 'unitDist_TT_hpm', 'highway','path_avg_bwness','path_avg_degree','adverse_weather']].describe())
        print('highway', self.data['highway'].value_counts())
        print('adverse_weather', self.data['adverse_weather'].value_counts())
    
    def save_data(self):
        train_df, test_df = train_test_split(self.data, test_size=0.20, random_state=42)
        print('length of train data {} and test data {}'.format(len(train_df), len(test_df)))
        data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/data/modeling_data/{}/'.format(self.data_type)
        train_df.to_csv(data_path + 'train.txt')
        test_df.to_csv(data_path + 'test.txt')
        with open(data_path + 'scale_parameters.pickle', 'wb') as handle:
            pickle.dump(self.scale_parameters, handle)