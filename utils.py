'''
Descripttion: 一些工具函数定义，如Bootstrap采样函数，shuffle函数和绘图函数
Version: 1.0
Date: 2021-04-03 16:31:02
LastEditTime: 2021-05-05 15:28:32
'''
import numbers
import numpy as np
import pandas as pd
import os
import  matplotlib.pyplot as plt
from scipy.sparse import issparse
from sklearn.metrics import roc_curve

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

# boostrap采样与返回包外估计样本
def resample(*arrays,
             replace=True,
             n_samples=None,
             random_state=None,
             ):

    max_n_samples = n_samples
    if len(arrays) == 0:
        return None

    random_state = check_random_state(random_state)
    first = arrays[0]
    n_samples = first.shape[0] if hasattr(first, 'shape') else len(first)

    if max_n_samples is None:
        max_n_samples = n_samples
    elif (max_n_samples > n_samples) and (not replace):
        raise ValueError("Cannot sample %d out of arrays with dim %d "
                         "when replace is False" % (max_n_samples,
                                                    n_samples))

    if replace: #有放回
        indices = random_state.randint(0, n_samples, size=(max_n_samples,))
    else:   #无放回
        indices = np.arange(n_samples)
        random_state.shuffle(indices)
        indices = indices[:max_n_samples]

    arrays = [a.tocsr() if issparse(a) else a for a in arrays]
    resampled_arrays = [a[indices] for a in arrays]
    
    # 获取包外估计样本下标
    all_indices = np.arange(n_samples)
    oof_indices = np.array(list(set(all_indices).difference(set(indices))))
    oof_arrays = [a[oof_indices] for a in arrays]
    
    if len(resampled_arrays) == 1:
        # syntactic sugar for the unit argument case
        return resampled_arrays[0], oof_arrays[0]
    else:
        return tuple(resampled_arrays), tuple(oof_arrays)

def sample_shuffle(X_train:pd.DataFrame, y_train:pd.Series):
    data = np.concatenate((X_train.to_numpy(), y_train.to_numpy().reshape(-1, 1)), axis=1)
    np.random.shuffle(data)
    X_cols, y_name = list(X_train.columns), y_train.name
    X_cols.append(y_name)
    df = pd.DataFrame(data, columns = X_cols)
    return df[X_train.columns], df[y_train.name]

def draw_roc_curve(y,  y_hat, test_roc_auc,  curve_save_direct, indus):
    fpr, tpr, thresholds = roc_curve(y, y_hat, pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, color='pink', label=f'AUC Bagging+DCRN = {test_roc_auc})')
    plt.legend(loc="lower right")
    plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), linestyle='--')
    plt.title( 'ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(os.path.join( curve_save_direct,  f'{indus}_k_fold_ROC_curve.png'))