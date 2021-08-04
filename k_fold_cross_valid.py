'''
Descripttion: k折交叉验证
Version: 1.0
Date: 2021-03-31 20:21:21
LastEditTime: 2021-05-05 15:28:23
'''
import config
import numpy as np
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from utils import draw_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from bagging import Bagging
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
# plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

def bagging_k_fold_cross_valid(X_train, y_train, indus):
    model_save_direct = os.path.join(config.model_root, indus, "dcrn_k_fold")
    final_model_save_direct = os.path.join(config.model_root, indus, "dcrn_final")
    if not os.path.exists(model_save_direct):
        os.makedirs(model_save_direct)
    if not os.path.exists(final_model_save_direct):
        os.makedirs(final_model_save_direct)

    config.dcrn_params[indus]['feature_size'] = X_train.shape[1]

    # K折交叉验证
    k = config.num_splits
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.random_seed)

    sum_test_acc, sum_test_f1, sum_test_roc_auc = 0.0, 0.0, 0.0

    y_hat_probas = []
    y_hat_categorys = []
    ys = []
    stop_epoch_sum = 0 # 存储k折的总迭代轮数，以得到平均迭代轮数
    for i, (train_idx, valid_idx) in enumerate(kfold.split(X_train, y_train)):

        print("第 %d 折交叉验证开始" % i)
        print(len(train_idx),"***", len(valid_idx))
        train_X, valid_X = X_train[train_idx], X_train[valid_idx]
        train_y, valid_y = y_train[train_idx], y_train[valid_idx]

        # SMOTE过采样
        # smo = SMOTE(random_state=42, n_jobs=-1 )
        # train_X,  train_y = smo.fit_resample(train_X, train_y)

        # 进行T次bootstrap采样，训练T个bagging基分类器
        bagging_model = Bagging(config.num_base_model[indus], **config.dcrn_params[indus])
        bagging_model.compile(**config.compiled_params[indus])
        config.fit_params[indus]['validation_data'] = (valid_X, valid_y)
        stop_epoch_sum += bagging_model.fit(train_X, train_y, **config.fit_params[indus])

        bagging_model.save(os.path.join(model_save_direct, f"{i}th_fold"))
        
        bagging_model.load(os.path.join(model_save_direct, f"{i}th_fold"))
        # 此处采用先将预测结果存起来最后再评估的方式
        # evaluate_res = bagging_model.evaluate(valid_X, valid_y, **config.evaluate_params)

        y_hat_proba = bagging_model.predict_proba(valid_X, **config.evaluate_params[indus])
        y_hat_probas.append(y_hat_proba)
        y_hat_category =bagging_model.predict(valid_X, **config.evaluate_params[indus])
        y_hat_categorys.append(y_hat_category)
        ys.append(valid_y)


    y_hat_proba = np.concatenate(tuple(y_hat_probas), axis=0)
    y_hat_category = np.concatenate(tuple(y_hat_categorys), axis=0)
    y = np.concatenate(tuple(ys), axis=0)
    test_roc_auc = roc_auc_score(y, y_hat_proba)
    acc_score = accuracy_score(y,  y_hat_category )
    stop_epoch_avg  = stop_epoch_sum//k
    print("k折平均 test acc %.3f,  test auc %.3f, 迭代次数:%d" % (acc_score, test_roc_auc, stop_epoch_avg))
    stop_epoch_data = pd.DataFrame({'times':[stop_epoch_avg]})
    stop_epoch_data.to_csv(os.path.join(final_model_save_direct, "迭代次数.csv"), index=False)
    curve_save_direct = os.path.join(config.roc_curve_root, indus)
    if not os.path.exists(curve_save_direct):
        os.makedirs(curve_save_direct)
    draw_roc_curve(y,  y_hat_proba,  test_roc_auc,  curve_save_direct, indus)





def bagging_train_final(X_train, y_train, indus):
    model_save_direct = os.path.join(config.model_root, indus, "dcrn_final")
    if not os.path.exists(model_save_direct):
        os.makedirs(model_save_direct)

    # 读取迭代次数信息
    stop_epoch_data = pd.read_csv(os.path.join(model_save_direct, "迭代次数.csv"))

    stop_epoch = stop_epoch_data['times'].to_list()[0]

    # SMOTE过采样
    #smo = SMOTE(random_state=42, n_jobs=-1 )

    # X_sampling,  y_sampling = smo.fit_resample(X_train, y_train)

    config.dcrn_params[indus]['feature_size'] = X_train.shape[1]

    #训练bagging分类器
    bagging_model = Bagging(config.num_base_model[indus], **config.dcrn_params[indus])
    bagging_model.compile(**config.compiled_params[indus])

    # 设置迭代步数并关闭早停
    config.fit_params[indus]["epochs"] = stop_epoch
    if  'callbacks' in config.fit_params[indus].keys():
        config.fit_params[indus].pop('callbacks')
    if 'validation_data' in config.fit_params[indus].keys():
        config.fit_params[indus].pop('validation_data') 
    bagging_model.fit(X_train,  y_train,** config.fit_params[indus])

    bagging_model.save(model_save_direct)
