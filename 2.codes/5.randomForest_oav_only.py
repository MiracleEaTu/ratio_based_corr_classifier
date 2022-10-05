import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import metrics
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
__author__ = 'Qijie Guan'

# DO NOT RUN THIS SCRIPT
# Based on the lack of samples, level 0 and level 1 were merged as good samples, 
#    Level 2 samples were marked as poor samples.


def z_score_trans(df_in):
    zscore_scaler = StandardScaler()
    data_transfered = zscore_scaler.fit_transform(df_in.values)
    df_out = pd.DataFrame(data_transfered, columns=df_in.columns.values, index=df_in.index.values)
    return df_out


def dataset_seperation(df_in):
    df_in = df_in.T
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
    print(X)
    print(Y)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.2, random_state=5)
    return X, Y, Xtrain, Xtest, Ytrain, Ytest


def randomForest_optimization(df_in, X, Y, Xtrain, Xtest, Ytrain, Ytest):
    df_in = df_in.T
    import warnings
    warnings.filterwarnings("ignore")
    rfc = RandomForestClassifier(random_state=5)
    kfold = KFold(n_splits=10)
    scoring_fnc = metrics.make_scorer(metrics.accuracy_score)
    # Set candidates for parameters.
    parameters = {'n_estimators': range(30,300,10),'max_depth':range(3,50,2),\
        'min_samples_leaf':[2,3,4,5,6,7]}
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
    df_grid_out.to_excel('../1.data_generated/5.1.rfc_results_oav_only.xlsx', index=True)
    # save features
    features = df_in.columns.values
    feature_importances = rfc_optimized.feature_importances_
    print('##########')
    print(features)
    print('##########')
    print(feature_importances)
    features_df = pd.DataFrame({'Features':features,'Importance':feature_importances})
    features_df.sort_values('Importance',inplace=True,ascending=False)
    features_df.to_excel('../1.data_generated/5.2.rfc_featureImportance_oav_only.xlsx', index=False)
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
    plt.savefig('../3.figures/5.1.rfc_oav_only.pdf')
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
    plt.savefig('../3.figures/5.2.Barplot_rfc_Feature_Importance_oav_only.pdf')
    plt.close()
    return 0


def main():
    df_datatable_raw = pd.read_excel('../1.data_generated/3.oav_data_removed_outliers_by_LOF.xlsx', index_col=0)
    df_datatable = z_score_trans(df_datatable_raw)
    # df_datatable_raw.T
    X, Y, Xtrain, Xtest, Ytrain, Ytest = dataset_seperation(df_in=df_datatable)
    features_df = randomForest_optimization(df_datatable, X, Y, Xtrain, Xtest, Ytrain, Ytest)
    plot(features_df, n=15)


if __name__ == '__main__':
    # Be very careful of this script.
    main()
    pass