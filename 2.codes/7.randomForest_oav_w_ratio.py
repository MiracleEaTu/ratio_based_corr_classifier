import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import metrics
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import argparse
__author__ = 'Qijie Guan'


def z_score_trans(df_in):
    zscore_scaler = preprocessing.StandardScaler()
    data_transfered = zscore_scaler.fit_transform(df_in.values)
    df_out = pd.DataFrame(data_transfered, columns=df_in.columns.values, index=df_in.index.values)
    return df_out


def cat_datasheets(df_oav, df_ratio):
    df_oav_zscore = z_score_trans(df_oav)
    df_ratio_zscore = z_score_trans(df_ratio)
    df_all = pd.concat([df_oav_zscore, df_ratio_zscore])
    print(pd.concat([df_oav, df_ratio]))
    # df_all.to_excel('../1.data_generated/9.zscored_oav_w_logged2ratio_20211209.xlsx', index=True)
    return df_all


def dataset_seperation(df_in):
    df_in = df_in.T
    # print(df_in.index.values)
    # 每行数据为一个样本
    # Good samples were index names including Level_0 or Level_1
    index_good = [names for names in df_in.index.values if 'RG' in names]
    df_good = df_in.loc[index_good]
    X_good = df_good.values
    # Poor samples were index names including Level_2
    index_poor = [names for names in df_in.index.values if 'RP' in names]
    df_poor = df_in.loc[index_poor]
    X_poor = df_poor.values
    # make datatable for random forest
    X = np.concatenate((X_good, X_poor), axis=0)
    Y = []
    for i in range(len(df_good)):
        Y.append(0)
    for i in range(len(df_poor)):
        Y.append(1)
    # print(X,Y)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.2, random_state=5)
    # print(X, Y, Xtrain, Xtest, Ytrain, Ytest)
    return X, Y, Xtrain, Xtest, Ytrain, Ytest


def randomForest_optimization(df_in, X, Y, Xtrain, Xtest, Ytrain, Ytest):
    # reference: https://blog.csdn.net/qq_30490125/article/details/80387414
    import warnings
    warnings.filterwarnings("ignore")
    rfc = RandomForestClassifier(random_state=5)
    kfold = KFold(n_splits=10)
    scoring_fnc = metrics.make_scorer(metrics.accuracy_score)
    # Set candidates for parameters.
    parameters = {'n_estimators': range(30,300,10),'max_depth':range(3,50,2),\
        'min_samples_leaf':[2,3,4,5,6,7]}
    # parameters = {'n_estimators': range(30,40,10),'max_depth':range(10,12,2),\
    #     'min_samples_leaf':[3,4]}
    # grid_rfc = GridSearchCV(rfc, parameters, scoring='f1_macro')
    grid_rfc = GridSearchCV(rfc, parameters, cv=kfold)
    # grid_rfc.fit(Xtrain, Ytrain)
    grid_rfc.fit(X, Y)
    print('Best Parameters for RandomForest are %s, Best score is %f.'%(str(grid_rfc.best_params_), grid_rfc.best_score_))
    for key in parameters.keys():
        print('%s: %d'%(key, grid_rfc.best_estimator_.get_params()[key]))
    # Train model by optimized param.
    rfc_optimized = RandomForestClassifier(n_estimators=grid_rfc.best_params_['n_estimators'],\
        max_depth=grid_rfc.best_params_['max_depth'], min_samples_leaf=grid_rfc.best_params_['min_samples_leaf'])
    print('test score: %f'%grid_rfc.best_estimator_.score(Xtest, Ytest))
    rfc_optimized.fit(Xtrain, Ytrain)
    # rfc_optimized.fit(X, Y) 
    pred = rfc_optimized.predict(Xtest)
    print(metrics.classification_report(pred,Ytest))
    df_grid_out = pd.DataFrame(grid_rfc.cv_results_).T
    df_grid_out.to_excel('../1.data_generated/7.1.rfc_results_oav_w_ratio.xlsx', index=True)
    # save features
    features = df_in.index.values
    feature_importances = rfc_optimized.feature_importances_
    features_df = pd.DataFrame({'Features':features,'Importance':feature_importances})
    features_df.sort_values('Importance',inplace=True,ascending=False)
    features_df.to_excel('../1.data_generated/7.2rfc_featureImportance_oav_w_ratio.xlsx', index=False)
    # updated 2021-07-22
    # export ROC curve
    
    r = rfc_optimized.score(Xtest, Ytest)
    predict_y_validation = rfc_optimized.predict(Xtest)
    probe_predict_y_validation = rfc_optimized.predict_proba(Xtest)
    predictions_validation = probe_predict_y_validation[:,1]
    fpr, tpr, _ =  roc_curve(Ytest, predictions_validation)
    roc_auc = auc(fpr, tpr)
    plt.title('ROC Validation')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()
    plt.savefig('../3.figures/7.1_ROC_rfc_oav_w_ratio.pdf')
    plt.close()
    
    return features_df


def plot(features_df, n=15):
    # Plot importances
    sns.set(rc={"figure.figsize": (8, 8)})
    ax = sns.barplot(features_df['Features'][:n], features_df['Importance'][:n],)
    plt.ylabel('Importance')
    # 数据可视化：柱状图
    sns.despine(bottom=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.figure.tight_layout()
    plt.savefig('../3.figures/9.2_classification_Barplot_RandomForest_Feature_Importance_oav_w_ratio_%s.pdf'%str(datetime.date.today()))
    plt.close()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-oav','--inputoavfile', required=False, help='The OAV file path')
    print('Start data concating')
    df_oav = pd.read_excel('../1.data_generated/2.data_transfered2oav.xlsx', index_col=0)
    df_ratio = pd.read_excel('../1.data_generated/4.log2ratios_generated_by_oavs.xlsx', index_col=0)
    df_dataset4rfc = cat_datasheets(df_oav=df_oav, df_ratio=df_ratio)
    X, Y, Xtrain, Xtest, Ytrain, Ytest = dataset_seperation(df_in=df_dataset4rfc)
    features_df = randomForest_optimization(df_dataset4rfc, X, Y, Xtrain, Xtest, Ytrain, Ytest)
    plot(features_df, n=15)
    pass 