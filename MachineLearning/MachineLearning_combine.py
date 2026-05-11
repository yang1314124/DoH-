import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data_path = "C:/Users/Administrator/Desktop/trafficdet-main/alg/CSVs/Total_CSVs/"
# data_path = "D:/PCAPs/"
# malicious_doh_df = pd.read_csv(data_path+"malicious_doh4.csv", low_memory=False, delimiter=',')
malicious_doh_df = pd.read_csv(data_path+"l2-malicious.csv", low_memory=False, delimiter=',')
benign_doh_df = pd.read_csv(data_path+"l2-benign.csv", low_memory=False, delimiter=',')
non_doh_df = pd.read_csv(data_path+"l1-nondoh.csv",low_memory=False,delimiter=',')
malicious_doh_df['Label'] = 2
benign_doh_df['Label'] = 1
non_doh_df['Label'] = 0
data_df = shuffle(pd.concat([non_doh_df, malicious_doh_df, benign_doh_df]))
data_df1 = data_df.fillna(0)
data_df2 = data_df1.drop(labels=['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp'], axis=1)
# print(data_df2.head(10))
# counts = data_df1['Label'].value_counts()
# print(counts)
X = data_df2.drop(['Label'], axis=1)
y = data_df2['Label'].values
# print(X.head())
#
scaler = StandardScaler()
X = scaler.fit_transform(X)
#
print(f"X shape:{X.shape} Y shape:{y.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
#
#
#
def model_acc_func(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    print('Test Accuracy : \033[32m \033[01m {:.5f}% \033[30m \033[0m'.format(acc*100))
    print(classification_report(y_test, y_pred,digits=5))
    cf_matrix = confusion_matrix(y_test, y_pred)
    print(cf_matrix)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,fmt= '0.5%')
    return model, acc
#
# #RandomForestClassifier


rfc_model, rfc_acc = model_acc_func(RandomForestClassifier(), X_train, y_train, X_test, y_test)
#
# #DecisionTreeClassifier
# dtc_model, dtc_acc = model_acc_func(DecisionTreeClassifier(), X_train, y_train, X_test, y_test)

