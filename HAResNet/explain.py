import shap
from alg.resnet.model import ResNet
import torch
from alg.krcnn.loader import doh_dataloader
from alg.krcnn.model import KAResNet
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
# 类别0：非doh流量
# 类别1：恶意doh流量
# 类别2：良性doh流量
# 初始化模型并加载权重
model = KAResNet(32, 3)
state_dict = torch.load("C:/Users/Administrator/Desktop/trafficdet-main/alg/model/krcnn-doh.pth")
model.load_state_dict(state_dict)
model.eval()

# 加载数据
train_loader, valid_loader, test_loader, data_len1 = doh_dataloader("C:/Users/Administrator/Desktop/trafficdet-main/alg/Dataset/dataset_doh3.csv", 3, batch_size=64)
all_test_features = []
all_test_labels = []

# 遍历 test_loader
for idx, (features, labels) in enumerate(test_loader):
    all_test_features.append(features)  # 将特征移动到 CPU
    all_test_labels.append(labels)      # 将标签移动到 CPU
# 合并所有特征和标签
all_test_features = torch.cat(all_test_features, dim=0)
all_test_labels = torch.cat(all_test_labels, dim=0)
print(all_test_features.shape)
num_samples = 1000  # 需要采样的数量
indices = torch.randperm(all_test_features.size(0))[:num_samples]  # 随机生成索引

# 查看结果的形状
# print(f"测试集特征的形状: {all_test_features.shape}")
# print(f"测试集标签的形状: {all_test_labels.shape}")
background = all_test_features[indices]
# print(background.shape)# 保留为 torch.Tensor 类型
# print(background.shape[0])
# 初始化 GradientExplainer
explainer = shap.GradientExplainer(model, background)
#计算 SHAP 值
shap_values = []
for i in tqdm(range(background.shape[0]), desc="Calculating SHAP values"):
    shap_value = explainer.shap_values(background[i:i+1])  # 逐个样本计算 SHAP 值
    shap_values.append(shap_value)
#
shap_values = np.array(shap_values)
shap_values = np.squeeze(shap_values,axis=1)
# print(shap_values.shape)
shap_values_avg = np.mean(shap_values, axis=-1)
approx_expected_value = model(background).mean().item()
# print(shap_values.shape)
# print(background.shape)
feature_names = ['fiat_mean', 'fiat_std', 'biat_mean','biat_std',
                 'diat_mean', 'diat_std', 'duration', 'fwin_mean', 'fwin_std'
                ,'bwin_mean','bwin_std', 'dwin_mean','dwin_std',
                 'fpnum', 'bpnum', 'dpnum', 'bfpnum_rate', 'fpnum_s',
                 'bpnum_s', 'dpnum_s', 'fpl_mean', 'fpl_std', 'fpl_medium', 'bpl_mean',
                 'bpl_std','bpl_medium', 'dpl_mean','dpl_std', 'dpl_medium',
                 'f_ht_len', 'b_ht_len', 'd_ht_len']
if isinstance(feature_names, list):
    feature_names = np.array(feature_names)
#1、特征值重要性图片
# plt.figure(figsize=(15, 10))
# plt.title("")
# shap.summary_plot(shap_values_avg[:, :, 0, 0],plot_type="bar", max_display=32, feature_names=feature_names, show=False)
# plt.savefig("C:/Users/Administrator/Desktop/trafficdet-main/alg/krcnn/explain/feature_importance_plot.png",bbox_inches='tight')
# plt.show()
# plt.close()
# fig, ax1 = plt.subplots(figsize=(12, 10), dpi=300)
# shap.summary_plot(shap_values_avg[:, :, 0, 0],background[:, :, 0, 0], feature_names=feature_names, plot_type="dot", show=False, color_bar=True)
# plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
# 获取共享的 y 轴
# ax1 = plt.gca()
# # 创建共享 y 轴的另一个图，绘制特征贡献图在顶部x轴
# ax2 = ax1.twiny()
# shap.summary_plot(shap_values_avg[:, :, 0, 0],background[:, :, 0, 0],feature_names=feature_names, plot_type="bar", show=False)
# plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
# ax2.axhline(y=13, color='gray', linestyle='-', linewidth=1)  # 注意y值应该对应顶部
# # 调整透明度
# bars = ax2.patches  # 获取所有的柱状图对象
# for bar in bars:
#     bar.set_alpha(0.2)  # 设置透明度
# # 设置两个x轴的标签
# ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=12)
# ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=12)
# # 移动顶部的 X 轴，避免与底部 X 轴重叠
# ax2.xaxis.set_label_position('top')  # 将标签移动到顶部
# ax2.xaxis.tick_top()  # 将刻度也移动到顶部
# # 设置y轴标签
# ax1.set_ylabel('Features', fontsize=12)
# plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
# plt.savefig("SHAP_combined_with_top_line_corrected.pdf", format='pdf', bbox_inches='tight')
# plt.show()
# plt.close()
# 2、单、多个样本解释
shap.initjs(),
# 单样本

# 假设 shap_values 和 background 是你的数据，且第一维是样本数量
# 随机选择一个样本的索引
approx_expected_value = approx_expected_value.numpy() if hasattr(approx_expected_value, 'numpy') else approx_expected_value
# 单样本中类别0
shap_values_np0 = shap_values[0, :, 0, 0, 0].numpy() if hasattr(shap_values[0, :, 0, 0, 0], 'numpy') else shap_values[0, :, 0, 0, 0]
background_np0 = background[0, :, 0, 0].numpy() if hasattr(background[0, :, 0, 0], 'numpy') else background[0, :, 0, 0]
#单样本中类别1
shap_values_np1 = shap_values[0, :, 0, 0, 1].numpy() if hasattr(shap_values[0, :, 0, 0, 1], 'numpy') else shap_values[0, :, 0, 0, 1]
background_np1 = background[0, :, 0, 0].numpy() if hasattr(background[0, :, 0, 0], 'numpy') else background[0, :, 0, 0]
#单样本中类别2
shap_values_np2 = shap_values[0, :, 0, 0, 2].numpy() if hasattr(shap_values[0, :, 0, 0, 2], 'numpy') else shap_values[0, :, 0, 0, 2]
background_np2 = background[0, :, 0, 0].numpy() if hasattr(background[0, :, 0, 0], 'numpy') else background[0, :, 0, 0]

#多样本(暂时没用)
# shap_values_np3 = shap_values[0:10, :, 0, 0, 0].numpy() if hasattr(shap_values[0:10, :, 0, 0, 0], 'numpy') else shap_values[0:50, :, 0, 0, :]
# background_np3 = background[0:50, :, 0, 0].numpy() if hasattr(background[0:50, :, 0,0 ], 'numpy') else background[0:50, :, 0, :]
# print(shap_values_np3.shape)
# print(background_np3.shape)
#生成 SHAP force plot
shap_force_plot1 = shap.force_plot(approx_expected_value, shap_values_np0, background_np0, feature_names=feature_names)
shap_force_plot2 = shap.force_plot(approx_expected_value, shap_values_np1, background_np1, feature_names=feature_names)
shap_force_plot3 = shap.force_plot(approx_expected_value, shap_values_np2, background_np2, feature_names=feature_names)
# shap_force_plot4 = shap.force_plot(approx_expected_value, shap_values_np3, background_np3, feature_names=feature_names)
#保存为 HTML
shap.save_html(f"C:/Users/Administrator/Desktop/trafficdet-main/alg/krcnn/explain/shap_force_plot_one_label0_5.html", shap_force_plot1),
shap.save_html(f"C:/Users/Administrator/Desktop/trafficdet-main/alg/krcnn/explain/shap_force_plot_one_lable1_5.html", shap_force_plot2),
shap.save_html(f"C:/Users/Administrator/Desktop/trafficdet-main/alg/krcnn/explain/shap_force_plot_one_label2_5.html", shap_force_plot3),
# shap.save_html("C:/Users/Administrator/Desktop/trafficdet-main/alg/krcnn/explain/shap_force_plot_multi.html", shap_force_plot4)


# 3、全局样本解释
# plt.figure(figsize=(15, 10))
# shap.summary_plot(shap_values[:,:,0,0,:], background[:,:,0,0],feature_names=feature_names)
# plt.savefig("C:/Users/Administrator/Desktop/trafficdet-main/alg/krcnn/explain/global_sample_plot.png",show=False)
# plt.show()
# plt.close()

#4、影响力解释（没什么用）
# shap_values_np4 = shap_values[:, :, 0, 0, 0].numpy() if hasattr(shap_values[:, :, 0, 0, 0], 'numpy') else shap_values[:, :, 0, 0, 0]
# background_np4 = background[:, :, 0, 0].numpy() if hasattr(background[:, :, 0, 0],'numpy') else background[:, :, 0, 0]
# plt.figure(figsize=(15, 10))
# shap.dependence_plot(feature_names[0], shap_values_np4, background_np4, feature_names=feature_names,show=False)
# plt.savefig("C:/Users/Administrator/Desktop/trafficdet-main/alg/krcnn/explain/dependence_plot.png", bbox_inches='tight')
# plt.show()
# plt.close()

#5、模型预测值的构成
# explaination = explainer.shap_values(background)
# first_sample_shap_values_label1 = explaination[0,:,:,:,0]#非doh
# first_sample_shap_values_label2 = explaination[0,:,:,:,1]#恶意doh
# first_sample_shap_values_label3 = explaination[0,:,:,:,2]#非恶意doh
# base_value = approx_expected_value  # 获取基线值
# sample_data = background.detach().numpy()
# explanation1 = shap.Explanation(values=first_sample_shap_values_label1,
#                                 base_values=base_value,
#                                 data=sample_data,
#                                 feature_names=feature_names)
# 绘制瀑布图
# plt.figure(figsize=(15, 10))
# shap.plots.waterfall(explanation1[:,0,0,0],max_display=32,show=False)
# plt.savefig("C:/Users/Administrator/Desktop/trafficdet-main/alg/krcnn/explain/waterfall_plot.png",bbox_inches='tight')
# plt.show()
# plt.close()

