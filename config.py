'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-24 09:33:40
LastEditors: ZhangHongYu
LastEditTime: 2022-04-19 19:42:34
'''
import tensorflow as tf


# 随机数种子
random_seed = 2021


# 数据存放目录定义
data_root = 'data'
# 用于特征选择的模型的目录
features_model_root = 'features_model'
# 用于保存模型特征重要性的目录
features_imp_root = 'features_imp'
# 用于保存模型预测结果目录
prediction_root = 'prediction'

top_n = 30

# 制造业bagging模型的基分类器个数，即boostrap采样次数
num_base_model_1 = 4
#非制造业bagging模型的基分类器个数，即boostrap采样次数
num_base_model_2 = 4
num_base_model = {
    '制造业':num_base_model_1,
    "其他行业":num_base_model_2
}

# 训练好的模型存放目录
model_root = 'model'

# 存放ROC曲线目录
roc_curve_root = 'roc_curve'



# 制造业模型的架构超参数
dcrn_params_1 = {
    "deep_layers": [56, 56, 56, 56], 
    "dropout_deep": [0,  0, 0,  0, 0],#深度网络的dropout，限定为len(deep_layers)+1
    "residual_layers": [36, 36, 36],#残差网络的隐藏层维度，不包括最后一层输出层
    "residual_blocks_num":6,#残差块个数，6个最佳
    "dropout_residual": [0, 0, 0, 0, 0], #残差网络的dropout，长度要为len(residual_layers)+2
    "residual_layers_activation":tf.nn.relu,
    "random_seed":random_seed,
    "deep_layers_activation": tf.nn.relu,
    "cross_layer_num":12
}
# 其他行业模型的架构超参数
dcrn_params_2 = {
    "deep_layers": [128, 128, 128, 128],  
    "dropout_deep": [0,  0, 0,  0, 0],#深度网络的dropout，限定为len(deep_layers)+1
    "residual_layers": [56, 56, 56],#残差网络的隐藏层维度，不包括最后一层输出层
    "residual_blocks_num":6,#残差块个数，6个最佳
    "dropout_residual": [0, 0, 0, 0, 0], #残差网络的dropout，长度要为len(residual_layers)+2
    "residual_layers_activation":tf.nn.relu,
    "random_seed":random_seed,
    "deep_layers_activation": tf.nn.relu,
    "cross_layer_num":12
}
dcrn_params={
    "制造业":dcrn_params_1,
    "其他行业":dcrn_params_2
}
# 交叉验证折数
num_splits = 4


#制造业模型的compiled方法损失函数及优化器选项，注意：loss函数为交叉熵损失不能改动
learning_rate_1 = 0.01
compiled_params_1={
    "optimizer":  tf.keras.optimizers.RMSprop(learning_rate=learning_rate_1),
    "loss": 'CategoricalCrossentropy',
    "metrics": ['BinaryAccuracy', 'AUC']
        # 早停回调函数
}

#非制造业模型的compiled方法损失函数及优化器选项，注意：loss函数为交叉熵损失不能改动
learning_rate_2 = 0.01
compiled_params_2={
    "optimizer":  tf.keras.optimizers.RMSprop(learning_rate=learning_rate_2),
    "loss": 'CategoricalCrossentropy',
    "metrics": ['BinaryAccuracy', 'AUC']
        # 早停回调函数
}
compiled_params={
    "制造业":compiled_params_1,
    "其他行业":compiled_params_2
}


# 制造业模型的fit方法超参数
fit_params_1 = {
    "epochs":20,
    "batch_size":1024,
    "verbose":2,
    "callbacks" : [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `AUC` is no longer improving
            monitor="val_AUC",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=0,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=0,
            verbose=2,
        )
    ]
}

# 非制造业模型的fit方法超参数
fit_params_2 = {
    "epochs":20,
    "batch_size":1024,
    "verbose":2,
    "callbacks" : [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `AUC` is no longer improving
            monitor="val_AUC",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=0,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=0,
            verbose=2,
        )
    ]
}

fit_params={
    '制造业': fit_params_1,
    "其他行业": fit_params_2
}

evaluate_params_1={
    "batch_size":1024,
    "verbose":2
}
evaluate_params_2={
    "batch_size":1024,
    "verbose":2
}
evaluate_params={
    '制造业': evaluate_params_1,
    "其他行业": evaluate_params_2
}