import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# 加载数据集
# 替换为你的实际数据集文件路径
data_path = "D:/PCAPs/"
malicious_data = pd.read_csv(data_path+"malicious_doh4.csv")
normal_data = pd.read_csv(data_path+"normal_doh4.csv")
malicious_data['label'] = malicious_data['Label']
malicious_data['label'] = "Malicious_DoH"
normal_data['label'] = normal_data['Label']
normal_data['label'] = normal_data['label'].replace({'normal': 'Non_DoH', 'DoH': 'Benign_DoH'})
data_df = pd.concat([normal_data, malicious_data])
numerical_features = data_df.select_dtypes(include=['int64', 'float64']).columns
# print(data_df.head())
print(data_df['label'].value_counts())


def plot_distribution_target(df, target, features):
    plt.figure(figsize=(10, 6))

    for feature in features:
        ax = sns.kdeplot(df[df[target] == "Malicious_DoH"][feature], color='Red', label='target=Malicious_DoH')
        ax = sns.kdeplot(df[df[target] == "Non_DoH"][feature], color='Blue', label='target=Non_DoH')
        ax = sns.kdeplot(df[df[target] == "Benign_DoH"][feature], color='Green', label='target=Benign_DoH')

        # Set x-axis limits to the min and max of the feature
        ax.set_xlim(df[feature].min(), df[feature].max())

        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.legend()

        plt.show()


plot_distribution_target(data_df, 'label', numerical_features)


