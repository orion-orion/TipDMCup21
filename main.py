'''
Descripttion: 项目主要流程和数据pipline
Version: 1.0
Date: 2021-03-24 09:33:40
LastEditTime: 2021-08-04 16:32:23
'''
import tensorflow as tf

import pandas as pd
import numpy as np
import config
from utils import sample_shuffle
from feature_eng import data_preprocess, feature_importance_rank
from k_fold_cross_valid import bagging_k_fold_cross_valid, bagging_train_final
from predict import bagging_predict

if __name__ == '__main__':
    train_data_dict, test_data_dict, test_ticker_symbol_dict = data_preprocess()


    #模型训练部分
    for indus, indus_data in train_data_dict.items():

        y_train  = indus_data['FLAG']

        X_train = indus_data.drop('FLAG', axis=1)

        X_train, y_train = sample_shuffle(X_train, y_train)

        #用模型对特征重要性进行排序并输出top30重要的特征
        feature_importance_rank(X_train, y_train, 'predict', indus)

        X_train, y_train = np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int64)


        # 采用交叉验证法对深度学习模型进行交叉验证和参数调优
        bagging_k_fold_cross_valid(X_train, y_train, indus)

        # 采用全部数据集对模型进行训练
        bagging_train_final(X_train, y_train, indus)


    #模型推断部分
    for indus, indus_data in test_data_dict.items():

        X_test = indus_data.drop('FLAG', axis=1)

        # 测试集不能进行shuffle，否则后期和股票编号的映射关系对不上

        X_test = np.array(X_test, dtype=np.float32)

        #预测题目要求的高送转结果
        bagging_predict(X_test,  test_ticker_symbol_dict[indus], indus)