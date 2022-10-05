import pandas as pd 
import numpy as np 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.cross_decomposition import PLSRegression
import datetime
from collections import OrderedDict


# 根据等级分类
def print_PCA(df_in, pca_file_name):
    x_raw = df_in.T.values
    x = np.nan_to_num(x_raw)
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)   
    names_raw = list(df_in.T.index.values)
    y = []
    for names in names_raw:
        y.append(names.split('.')[0])
    # print(y)
    principleDF = pd.DataFrame(data=principalComponents, columns=['PC1','PC2'])
    principleDF['group'] = y
    finalDF = principleDF
    # print(principleDF)
    fig = plt.figure(figsize=(8, 8))
    plt.rcParams['font.family']=['Arial']
    ax = fig.add_subplot(1, 1, 1)
    print(type(pca.explained_variance_ratio_[0]))
    print(len(pca.explained_variance_ratio_))
    ax.set_xlabel('PC1 (%.2f %%)'%(pca.explained_variance_ratio_[0]*100), fontsize=15)
    ax.set_ylabel('PC2 (%.2f %%)'%(pca.explained_variance_ratio_[1]*100), fontsize=15)
    # color = {'Excellent grade':'#FF9966', 'Grade I':'#FFFFCC', 'Grade II':'#0066CC'}
    color = {'RG':'#FF9966', 'RP':'#FFFFCC'}
    for i in range(len(finalDF)):
        ax.scatter(finalDF.loc[i, 'PC1'], finalDF.loc[i,'PC2'],\
            c=color[finalDF.loc[i, 'group']], marker='o',\
                alpha=0.7, edgecolor='black', s=60, label=y[i])
    # for i in range(len(finalDF)):
    #     ax.annotate(names_raw[i], xy=(finalDF.loc[i, 'PC1'], finalDF.loc[i,'PC2']),\
    #         xytext=(finalDF.loc[i, 'PC1']+0.1, finalDF.loc[i,'PC2']+0.1), fontproperties="SimHei")
    # groups = ['优级', '一级', '二级']
    # plt.rcParams['font.family']=['SimHei']
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    # ax.legend()
    ax.grid(which='both')
    plt.savefig('../3.figures/%s.pdf'%(pca_file_name))
    return 0


def remove_outliers(df_in):
    df_using = df_in.T
    df_level_0 = df_using.loc[[indexes for indexes in df_using.index.values if 'RG' in indexes]]
    df_level_1 = df_using.loc[[indexes for indexes in df_using.index.values if 'RP' in indexes]]
    X_level_0 = df_level_0.values
    X_level_1 = df_level_1.values
    clf = LOF(n_neighbors=2)
    res_level_0 = clf.fit_predict(X=X_level_0)
    res_level_1 = clf.fit_predict(X=X_level_1)
    print(res_level_0)
    print(res_level_1)
    index_level0_remaining = [index_name for index_name, res_value in zip(df_level_0.index.values, res_level_0) if res_value == 1]
    df_level0_remaining = df_level_0.loc[index_level0_remaining]
    index_level1_remaining = [index_name for index_name, res_value in zip(df_level_1.index.values, res_level_1) if res_value == 1]
    df_level1_remaining = df_level_1.loc[index_level1_remaining]
    df_remaining = df_level0_remaining.append(df_level1_remaining)
    return df_remaining.T


# def PLS_using_sklearn(df_in):
#     df_using = df_in.T
#     Y = []
#     colors = []
#     names_raw = list(df_using.index.values)
#     # 定义优级为(-1, 1)，一级(-1, -1)，二级(1, 0)
#     ## 优级与一级相对二级差异更显著
#     for index_names in list(df_using.index.values):
#         if '优级' in index_names:
#             Y.append([-1, 1])
#             colors.append('red')
#         elif '一级' in index_names:
#             Y.append([-1, -1])
#             colors.append('blue')
#         elif '二级' in index_names:
#             Y.append([1, 0])
#             colors.append('gray')
#         else:
#             print('error. 分级错误，请检查原始数据集')
#     X = df_using.values
#     pls_modeling = PLSRegression(n_components=10)
#     pls_modeling.fit(X, Y)
#     Y_pred = pls_modeling.predict(X)
#     fig = plt.figure(figsize=(8, 8))
#     plt.rcParams['font.family']=['Arial']
#     ax = fig.add_subplot(1, 1, 1)
#     for i in range(len(Y_pred)):
#         ax.scatter(Y_pred[i][0], Y_pred[i][1], c=colors[i], marker='o', alpha=0.7, edgecolor='black', s=60)
#     for i in range(len(Y_pred)):
#         ax.annotate(names_raw[i], xy=(Y_pred[i][0], Y_pred[i][1]),\
#             xytext=(Y_pred[i][0]+0.1, Y_pred[i][1]+0.1), fontproperties="SimHei")
#     ax.grid(which='both')
#     plt.savefig('../3.figures/PLS_linerRegression_%s.png'%(str(datetime.date.today())))
#     return 0


if __name__ == '__main__':
    path_all = os.listdir('../')
    if '1.data_generated' not in path_all:
        os.mkdir('../3.figures/')
    df_in = pd.read_excel('../1.data_generated/0.datasheet_filtered_by_nonzeros.xlsx', index_col=0)
    print_PCA(df_in=df_in, pca_file_name='0.PCA_datasheet')
    df_in_transformed = pd.read_excel('../1.data_generated/2.data_transfered2oav.xlsx', index_col=0)
    print_PCA(df_in=df_in_transformed, pca_file_name='1.PCA_OAV_datasheet')
    df_remaining = remove_outliers(df_in=df_in_transformed)
    df_remaining.to_excel('../1.data_generated/3.oav_data_removed_outliers_by_LOF.xlsx', index=True)
    print_PCA(df_in=df_remaining, pca_file_name='2.PCA_OAV_datasheet_removed_outliers')
    # PLS_using_sklearn(df_in=df_remaining)
