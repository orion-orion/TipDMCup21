import pandas as pd
import numpy as np
from sklearn import preprocessing
data = pd.read_csv("preprocessed_data.csv")
# cols = ['ACT_PUBTIME', 'PUBLISH_DATE', 'END_DATE_REP', 'END_DATE']
# for col in cols:
#     # 对数值特征z-score标准化
#     scaler = preprocessing.StandardScaler().fit(
#         np.array(data[col]).reshape(-1, 1))
#     result = scaler.transform(np.array(data[col]).reshape(-1, 1))  
#     data.loc[:, col] = result 
#     # 对数值特征二范数归一化，该操作独立对待样本，无需对normalizer进行fit
#     # 但dummy编码不好处理，故不考虑之
#     # data.loc[:, col] = preprocessing.normalize(np.array(data[col]).reshape(-1, 1),norm='l2')
# data.to_csv("preprocessed_data.csv", index=False)
print(data)