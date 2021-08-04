'''
Descripttion: 负责调用最终的Bagging模型(已用所有数据训练)预测第6年的数据
Version: 1.0
Date: 2021-03-31 20:21:21
LastEditTime: 2021-05-05 15:26:59
'''
import config
import os
from bagging import Bagging
import pandas as pd
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
# plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

def bagging_predict(X_test,  test_ticker_symbol,  indus):
    model_save_direct = os.path.join(config.model_root, indus, "dcrn_final")
    prediction_direct = os.path.join(config.prediction_root, indus)
    if not os.path.exists(model_save_direct):
        raise IOError(f"Cant find the path {model_save_direct}")
    if not os.path.exists(prediction_direct):
       os.makedirs(prediction_direct)

    config.dcrn_params[indus]['feature_size'] = X_test.shape[1]

    #加载bagging分类器
    bagging_model = Bagging(config.num_base_model[indus], **config.dcrn_params[indus])
    bagging_model.compile(**config.compiled_params[indus])
    bagging_model.load(model_save_direct)

    y_pred = bagging_model.predict(X_test, ** config.evaluate_params[indus])
    predict_data = pd.DataFrame({"股票代码":test_ticker_symbol, "预测值":y_pred})
    pos_predict = predict_data[predict_data["预测值"]==1]['股票代码']
    pos_predict.to_csv(os.path.join(prediction_direct, f"{indus}_prediction.csv"), index=False)
