import numpy as np
import pandas as pd
from preprocess_data import *
# from model_HBR import *
# from model_ngboost import *
# from linearRegression import *
# from randomForest import *
# from lightGBM import *
# from model_evaluation import *

if __name__ == '__main__':
    # process raw data : note data should have the shortest path info already

    dp = data_processing(data_type='taxi')
    # exit()
    # hbr = HBR_pymc(data_type='amb')
    # ng = ngboost_(data_type='amb')
    # LR = LRegression(data_type='taxi')
    # RF = RForest(data_type='amb')
    # LGBM = lightGBM(data_type='amb')
    eval = models_comparison(data_type='taxi')

    # data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/HBRegression/TT_pred_wHBR/data/raw_data/{}/data.txt'.format('taxi')
    # data = pd.read_csv(data_path, index_col=0, dtype={'OriginNodes' : str, 'DestNodes' : str}).rename(columns={'From':'Oatom', 'To':'Datom'})
    # print(data)


