'''
Descripttion: 特征工程相关内容，包括数据预处理和特征选择等
Version: 1.0
Date: 2021-02-18 13:15:08
LastEditTime: 2022-04-19 19:36:08
'''
import pandas as pd
import config
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn import model_selection
import joblib

# 用于特征选择的模型定义
models={}


models.update({'dt':
    DecisionTreeClassifier(random_state=0)
})
models.update({'rf': 
    RandomForestClassifier(random_state=0)
})
models.update({'et': 
    ExtraTreesClassifier(random_state=0)
})
models.update({'xgb': 
    XGBClassifier(random_state=0)
})


# 用于特征选择的模型的超参数搜索范围定义
param_grids = {}
param_grids.update({
    'dt':
    { 'min_samples_split': [2, 4], 'max_depth': [12]}
})
param_grids.update({
    'rf':
    {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [12],'n_jobs':[-1]}
})
param_grids.update({
    'et':
   {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [12],'n_jobs':[-1]}
})
param_grids.update({
    'xgb':
   {'n_estimators': [500],  'max_depth': [2], 'objective':['binary:logistic'], 'eval_metric':['logloss'],'use_label_encoder':[False],'nthread':[-1]}
})

 #  完成超参数网格搜索后的模型
model_grids={}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
for name,  param in param_grids.items():
    model_grids[name] = model_selection.GridSearchCV(models[name], param, n_jobs=-1, cv=kfold, verbose=1,scoring='f1')

def read_data():
    # 原始样本数据
    data = pd.read_csv(os.path.join(config.data_root, '附件2.csv'))
    return data

def get_data_dict(data1):
    # 各行业和股票代码对应的关系
    data2 = pd.read_csv(os.path.join(config.data_root, '附件1.csv'))
    industries = list(set(list(data2['所属行业'])))
    all_data = {}
    for indus in industries:
        indus_data =  data1[ data1['TICKER_SYMBOL'].isin(list(data2[data2['所属行业']==indus]['股票代码']))]
        if indus == '制造业':
            all_data[indus] = indus_data
        else:
            if "其他行业" not in all_data.keys():
                all_data["其他行业"] = pd.DataFrame()
            all_data["其他行业"] =  all_data["其他行业"].append(indus_data, ignore_index=True)
    return all_data

def get_ticker_symbol_dict(data1):
    # 各行业和股票代码对应的关系
    data2 = pd.read_csv(os.path.join(config.data_root, '附件1.csv'))
    industries = list(set(list(data2['所属行业'])))
    all_ticker_symbol = {}
    for indus in industries:
        indus_data =  data1[ data1['TICKER_SYMBOL'].isin(list(data2[data2['所属行业']==indus]['股票代码']))]
        if indus == '制造业':
            all_ticker_symbol[indus] = indus_data['TICKER_SYMBOL'].to_list()
        else:
            if "其他行业" not in all_ticker_symbol.keys():
                all_ticker_symbol["其他行业"] = []
            all_ticker_symbol["其他行业"].extend(indus_data['TICKER_SYMBOL'].to_list())
    return all_ticker_symbol



#用模型对特征重要性进行排序，挖掘top-30的重要特征
def feature_importance_rank(X, y, indus, mod):

    features_model_save_direct = os.path.join(config.features_model_root, indus)
    features_imp_save_direct = os.path.join(config.features_imp_root, indus)


    if not os.path.exists(features_model_save_direct):
        os.makedirs(features_model_save_direct)
        
    if not os.path.exists(features_imp_save_direct):
        os.makedirs(features_imp_save_direct)

    avg_feature_imp = {}
    if mod == 'retrain': #如果是对训练集进行特征选择
        # SMOTE过采样
        smo = SMOTE(random_state=42, n_jobs=-1 )
        X_sampling,  y_sampling = smo.fit_resample(X, y)

        #用所有数据训练用于特征选择的模型
        for name, _  in model_grids.items():
                # 这里才对model_grids[name]进行实际修改
                model_grids[name].fit(X_sampling, y_sampling)
                joblib.dump(model_grids[name], os.path.join(features_model_save_direct, name +'.json'))
                print(" features selection model %s has been trained " % (name))
                model_grid = model_grids[name]
                features_imp_sorted = pd.DataFrame({'feature': list(X),
                                                    'importance': model_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
                features = features_imp_sorted['feature'].to_list()
                features_imp =  features_imp_sorted['importance'].to_list()
                for feature, imp in zip(features, features_imp):
                    if feature not in avg_feature_imp.keys():
                        avg_feature_imp.update({feature:imp})
                    else:
                        avg_feature_imp[feature] += imp
                features_top_n =  features_imp_sorted.head(config.top_n)['feature']
                features_top_n_imp =  features_imp_sorted.head(config.top_n)['importance']
                features_output = pd.DataFrame({'features_top_n':features_top_n, 'importance':features_top_n_imp})
                features_output.to_csv(os.path.join(features_imp_save_direct, name+'_top_n_features_importance.csv'), index=False)


    elif mod == "load":
        # 加载用于特征选择的模型并选出top-30的特征
        for name, _ in model_grids.items():
            features_model_save_path = os.path.join(config.features_model_root, indus, name+'.json')
            if not os.path.exists(features_model_save_path):
                raise IOError("Cant find the path %s!" % features_model_save_path)
            
            model_grids[name] = joblib.load(features_model_save_path) 
            model_grid = model_grids[name]
            features_imp_sorted = pd.DataFrame({'feature': list(X),
                                                'importance': model_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
            features = features_imp_sorted['feature'].to_list()
            features_imp =  features_imp_sorted['importance'].to_list()
            for feature, imp in zip(features, features_imp):
                if feature not in avg_feature_imp.keys():
                    avg_feature_imp.update({feature:imp})
                else:
                    avg_feature_imp[feature] += imp
            features_top_n =  features_imp_sorted.head(config.top_n)['feature']
            features_top_n_imp =  features_imp_sorted.head(config.top_n)['importance']
            features_output = pd.DataFrame({'features_top_n':features_top_n, 'importance':features_top_n_imp})
            features_output.to_csv(os.path.join(features_imp_save_direct, name+'_top_n_features_importance.csv'), index=False)
    else:
        raise IOError("invalid mod!") 
    
    # 对所有模型预测的结果做平均
    for feature in avg_feature_imp.keys():
        avg_feature_imp[feature] /= 4

    avg_feature_imp = dict(sorted(avg_feature_imp.items(), key= lambda item : -item[1])[:config.top_n])
    avg_features_output = pd.DataFrame({'features_top_n':avg_feature_imp.keys(), 'importance':avg_feature_imp.values()})
    avg_features_output.to_csv(os.path.join(features_imp_save_direct, 'top_n_avg_features_importance.csv'), index=False)


def data_preprocess():

    # data = read_data()
    # #  获得每个特征的缺失信息
    # null_info = data.isnull().sum(axis=0)
    # #  丢弃缺失值多于30%的特征
    # features = [k for k, v in dict(null_info).items() if v < data.shape[0]* 0.3]
    # data = data[features]

    # null_info = data.isnull().sum(axis=0)

    # # 选去出需要填补缺失值的特征
    # features_fillna = [k for k, v in dict(null_info).items() if v > 0]
    # # 测试样本的'FLAG'本身缺失，不予填充
    # if 'FLAG' in features_fillna:
    #     features_fillna.remove('FLAG') 

    # # 剔除全0的特征和明显无用的特征，股票编号最后删除．取出后所有特征皆为数值型
    # drop_list = ['FISCAL_PERIOD', 'MERGED_FLAG',
    # ]
    # data = data.drop(drop_list, axis=1)

    # for ticker_symbol in list(set(data['TICKER_SYMBOL'].to_list())):
    #     # 对缺失值进行填补，每一个股票代码分别插值填补
    #     for feature in features_fillna:
    #         # 如果是非数值型特征或者是整型离散数值，用众数填补
    #         # mode()函数将列按出现频率由高到低排序，众数即第一行
    #         if str(data[data['TICKER_SYMBOL']==ticker_symbol][feature].dtype) == 'object' or str(data[data['TICKER_SYMBOL']==ticker_symbol][feature].dtype) =='int64':
    #             print(feature, data[data['TICKER_SYMBOL']==ticker_symbol][feature])
    #             data.loc[data['TICKER_SYMBOL']==ticker_symbol,  feature] = data[data['TICKER_SYMBOL']==ticker_symbol][feature].fillna(
    #                 data[data['TICKER_SYMBOL']==ticker_symbol][feature].mode().iloc[0]
    #             )
    #         #浮点连续数值型特征插值填补+平均数处理边缘值，如果全为nan则补0
    #         else:
    #             #先将中间的数据插值处理
    #             data.loc[data['TICKER_SYMBOL']==ticker_symbol,  feature] = data[data['TICKER_SYMBOL']==ticker_symbol][feature].interpolate( method="zero", axis=0, limit_direction='both')
    #             #边缘部分填充平均数
    #             data.loc[data['TICKER_SYMBOL']==ticker_symbol,  feature] = data[data['TICKER_SYMBOL']==ticker_symbol][feature].fillna(
    #                 data[data['TICKER_SYMBOL']==ticker_symbol][feature].mean()
    #             )
    #             #如果全为nan，则统一填充零
    #             data.loc[data['TICKER_SYMBOL']==ticker_symbol,  feature] = data[data['TICKER_SYMBOL']==ticker_symbol][feature].fillna(
    #                 0
    #             )
    #             if np.isnan(data[data['TICKER_SYMBOL']==ticker_symbol][feature]).any():
    #                 print(data[data['TICKER_SYMBOL']==ticker_symbol][feature])    

    # # 数值归一化
    # for col in data.columns:
    #     if col == 'FLAG' or col=='TICKER_SYMBOL': #跳过标签列和股票编号列
    #         continue
    #     elif str(data[col].dtype) == 'object':
    #         data = data.drop(col, axis=1)
    #     else:
    #         # 对数值特征z-score标准化
    #         scaler = preprocessing.StandardScaler().fit(
    #             np.array(data[col]).reshape(-1, 1))
    #         result = scaler.transform(np.array(data[col]).reshape(-1, 1))  
    #         data.loc[:, col] = result 

    preprocessed_data_path = os.path.join(config.data_root, 'preprocessed_data.csv')

    #data = data.astype({'TICKER_SYMBOL': 'int64'})
    #data.to_csv(preprocessed_data_path, index=False)

    if not os.path.exists(preprocessed_data_path):
        raise IOError(f"Cant find the path {preprocessed_data_path}")
    data = pd.read_csv(preprocessed_data_path)


    # 分离训练集和测试集
    train_data = data[np.isfinite(data['FLAG'].to_numpy())]
    test_data = data[np.isnan(data['FLAG'].to_numpy())]
    train_data_dict = get_data_dict(train_data)
    test_data_dict = get_data_dict(test_data)
    test_ticker_symbol_dict = get_ticker_symbol_dict(test_data)

    # train_data 和test_data均需要去掉股票编码
    for indus in train_data_dict.keys():
        train_data_dict[indus] = train_data_dict[indus].drop('TICKER_SYMBOL', axis=1)

    for indus in test_data_dict.keys():
        test_data_dict[indus] = test_data_dict[indus].drop('TICKER_SYMBOL', axis=1)
    return train_data_dict, test_data_dict, test_ticker_symbol_dict
