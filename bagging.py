'''
Descripttion: Bagging模型定义
Version: 1.0
Date: 2021-03-31 19:26:59
LastEditTime: 2021-05-05 15:25:04
'''
from DCRN import DCRN
import os
import config
import numpy as np
import tensorflow as tf
from utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#注意Bagging虽然采用类似keras定义的model接口，但并不继承于keras.model
class Bagging(object):
    def __init__(self, num_base_model, **base_model_params):
        super(Bagging, self).__init__()
        self.num_base_model = num_base_model
        self.base_models = [ DCRN(**base_model_params) for t in range(self.num_base_model)]

    def compile(self, optimizer='SGD', loss='CategoricalCrossentropy', metrics=[ 'BinaryAccuracy', 'AUC', 'F1-Score']):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        # base Model的metric直接在模型内自定义，此处不用手工compile参数
        # keras没有内置F1-Score评判标准，用Recall和Precision
        # Bagging的metric中的F1-score单独计算，我们只需计算base_model的Recall和Precision
        recall_metric= tf.keras.metrics.Recall(
            thresholds=0.5, top_k=None, class_id=None, name='Recall', dtype=None
        )
        precision_metric = tf.keras.metrics.Precision(
            thresholds=0.5, top_k=None, class_id=None, name='Precision', dtype=None
        )
        binary_accuracy_metric= tf.keras.metrics.BinaryAccuracy(
            name='binary_accuracy', dtype=None, threshold=0.5
        )
        auc_metric= tf.keras.metrics.AUC(
            num_thresholds=200, curve='ROC',
            summation_method='interpolation', name='AUC', dtype=None,
            thresholds=None, multi_label=False, label_weights=None
        )
        base_models_metrics = []
        if 'F1-Score' in metrics:
            base_models_metrics.append(recall_metric)
            base_models_metrics.append(precision_metric)
        if 'BinaryAccuracy' in metrics:
            base_models_metrics.append(binary_accuracy_metric)
        if 'AUC' in metrics:
            base_models_metrics.append(auc_metric)
        for t in range(self.num_base_model):
            self.base_models[t].compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=base_models_metrics)

    def fit(self, train_X, train_y,  epochs=30, batch_size=4, verbose=2, callbacks=None, class_weight=None, validation_data=None):
        stop_epoch_sum = 0  # 存储所有基分类器的总迭代轮数，以返回平均迭代轮数
        for t in range(self.num_base_model):
            print(f"训练第{t}个基分类器")
            train_data, oof_data = resample(
                train_X, train_y, n_samples=train_X.shape[0], random_state=config.random_seed)
            train_X_t, train_y_t = train_data
            oof_X_t, oof_y_t = oof_data
            # 用boostrap抽得样本做训练
            history = self.base_models[t].fit(
                train_X_t, train_y_t, epochs=epochs, batch_size=batch_size,
                verbose=verbose, callbacks=callbacks, class_weight=class_weight, validation_data=validation_data
            )
            if 'callbacks' in config.fit_params['制造业'].keys() and 'callbacks' in config.fit_params['其他行业'].keys(): #如果是交叉验证模式，则跟踪迭代次数 
                stop_epoch_sum += len(history.history['val_AUC']) 
        return stop_epoch_sum//self.num_base_model
          
    def evaluate(self, test_X, test_y, batch_size=64, verbose=2):
        y_hat_list, y_hat_category_list = [], []
        for t in range(self.num_base_model):
            # evaluate模式关闭drop
            self.base_models[t].dropout_deep= [0] * len(self.base_models[t].dropout_deep)
            print(f"测试集对第{t}个基分类器的测试结果:")
            self.base_models[t].evaluate(test_X, test_y, batch_size=batch_size, verbose=verbose)
            y_hat_t = self.base_models[t].predict(test_X, batch_size=batch_size, verbose=verbose)
            y_hat_category = np.logical_not(np.less(y_hat_t, 0.5)).astype(np.int64)
            y_hat_list.append(y_hat_t.reshape(-1, 1))
            y_hat_category_list.append(y_hat_category.reshape(-1, 1))
        y_hat_mean = np.mean(np.concatenate(tuple(y_hat_list), axis=1), axis=1)
        y_hat_category = np.concatenate(tuple(y_hat_category_list), axis=1)
        # 总结0类和1类的投票表决计数
        y_hat_category_cnt = np.concatenate((np.sum((y_hat_category==0).astype(np.int32), axis=1).reshape(-1, 1), np.sum((y_hat_category==1).astype(np.int32), axis=1).reshape(-1, 1)), axis=1)
        # 多数投票表决法
        y_hat_category = np.argmax(y_hat_category_cnt, axis=1)
        evaluate_res = {}
        if 'BinaryAccuracy' in self.metrics:
            acc_score = accuracy_score(test_y,  y_hat_category )
            evaluate_res.update({'BinaryAccuracy':acc_score})
        if 'F1-Score' in self.metrics:
            test_f1 =  f1_score(test_y,  y_hat_category )
            evaluate_res.update({'F1-Score':test_f1})
        if 'AUC' in self.metrics:
            test_roc_auc = roc_auc_score(test_y, y_hat_mean)
            fpr, tpr, thresholds = roc_curve(test_y, y_hat_mean, pos_label=1)
            evaluate_res.update({'FPR':fpr})
            evaluate_res.update({'TPR':tpr})
            evaluate_res.update({'AUC':test_roc_auc})
        return evaluate_res

    def predict(self, X, batch_size=64, verbose=2):
        y_hat_category_list = []
        for t in range(self.num_base_model):
            # predict模式关闭drop
            self.base_models[t].dropout_deep= [0] * len(self.base_models[t].dropout_deep)
            y_hat_t = self.base_models[t].predict(X,batch_size=batch_size, verbose=verbose)
            y_hat_category = np.logical_not(np.less(y_hat_t, 0.5)).astype(np.int64)
            y_hat_category_list.append(y_hat_category.reshape(-1, 1))
        y_hat_category = np.concatenate(tuple(y_hat_category_list), axis=1)
        # 总结0类和1类的投票表决计数
        y_hat_category_cnt = np.concatenate((np.sum((y_hat_category==0).astype(np.int32), axis=1).reshape(-1, 1), np.sum((y_hat_category==1).astype(np.int32), axis=1).reshape(-1, 1)), axis=1)
        # 多数投票表决法
        y_hat_category = np.argmax(y_hat_category_cnt, axis=1)
        return y_hat_category

    def predict_proba(self, X, batch_size=64, verbose=2):
        y_hat_list = []
        for t in range(self.num_base_model):
            # evaluate模式关闭drop
            self.base_models[t].dropout_deep= [0] * len(self.base_models[t].dropout_deep)
            y_hat_t = self.base_models[t].predict(X, batch_size=batch_size, verbose=verbose)
            y_hat_list.append(y_hat_t.reshape(-1, 1))
        y_hat_mean = np.mean(np.concatenate(tuple(y_hat_list), axis=1), axis=1)
        return y_hat_mean

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for t in range(self.num_base_model):
            base_model_save_path = os.path.join(save_path, f"{t}th_base_model")
            if not os.path.exists(base_model_save_path):
                os.makedirs(base_model_save_path)
            self.base_models[t].save(base_model_save_path) 
    
    def load(self, load_path):
        if not os.path.exists(load_path):
            raise IOError(f"Can't find the path {load_path} !")
        for t in range(self.num_base_model):
            base_model_load_path = os.path.join(load_path, f"{t}th_base_model")
            if not os.path.exists(base_model_load_path):
                raise IOError(f"Can't find the {t}th base model!")
            self.base_models[t] = tf.keras.models.load_model(base_model_load_path)
            
                
            



        
            

        