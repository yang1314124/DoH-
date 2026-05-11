import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
def precess_nan_and_scaler(df):
    # 处理缺失值
    df.fillna(0, inplace=True)

    # 规范化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.values)
    # 将标准化后的数据转换回 DataFrame
    df_scaled = pd.DataFrame(data_scaled, columns=df.columns)

    return df_scaled

data_path = "D:/PCAPs/"
data_path2 = "C:/Users/Administrator/Desktop/trafficdet-main/alg/"
malicious_doh_df = pd.read_csv(data_path+"malicious_doh4.csv", low_memory=False)
non_doh_df = pd.read_csv(data_path+"normal_doh4.csv", low_memory=False)
label_distribution = non_doh_df['Label'].value_counts()
print(label_distribution)
#print(non_doh_df.head())
malicious_doh_df['Label'] = 2
non_doh_df['Label'] = non_doh_df['Label'].replace({'normal': 0, 'DoH': 1})
malicious_doh_df = malicious_doh_df.drop(0, axis=0)
non_doh_df = non_doh_df.drop(0, axis=0)
data_doh_df = pd.concat([malicious_doh_df, non_doh_df])
Label = data_doh_df.iloc[:, -1]
data_doh_df2 = precess_nan_and_scaler(data_doh_df.iloc[:, :-1])
Label = Label.reset_index(drop=True)
# 添加列
data_doh_df2[Label.name] = Label
#print(data_doh_df2.head())
label_distribution = data_doh_df2['Label'].value_counts()
print(label_distribution)
data_doh_df2.to_csv(data_path2+"Dataset/dataset_doh3.csv", index=False)
#
# # 检查保存的数据
# # 验证是否保存成功
try:
    _ = pd.read_csv(data_path2+"Dataset/dataset_doh3.csv")
    print(data_doh_df2.shape)
except Exception as e:
    print(f"An error occurred: {e}")

print('Done!')