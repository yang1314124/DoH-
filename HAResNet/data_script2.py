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
data_path = "C:/Users/Administrator/Desktop/trafficdet-main/alg/"
data_path2 = "D:/PCAPs/"
malicious_df = pd.read_csv(data_path2+"malicious_doh4.csv", low_memory=False)
#print(dns2tcp_df.head())
malicious_df['Label'] = malicious_df['Label'].replace({'dns2tcp': 0, 'dnscat2': 1,'iodine':2})
# print(malicious_df.head())

Label = malicious_df.iloc[:, -1]
malicious_df2 = precess_nan_and_scaler(malicious_df.iloc[:, :-1])
Label = Label.reset_index(drop=True)
# 添加列
malicious_df2[Label.name] = Label


malicious_df2.to_csv(data_path+"Dataset/dataset_malicious_doh3.csv", index=False)

try:
    _ = pd.read_csv(data_path+"Dataset/dataset_malicious_doh3.csv")
    print(malicious_df2.shape)
except Exception as e:
    print(f"An error occurred: {e}")

print('Done!')