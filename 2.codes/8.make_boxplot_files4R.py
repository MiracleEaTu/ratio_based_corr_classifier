import pandas as pd 
import numpy as np 
import argparse
# from rpy2.robjects.packages import importr
__author__ = 'Qijie Guan'


def plot_ratios_violin(file_path, listOfRatios=[], df_print_name=''):
    # df_golden_ratios_all = pd.read_excel('../1.data_generated/5.good_ratios_log2_all_samples_2021-07-05.xlsx', index_col=0)
    df_ratios = pd.read_excel(file_path, index_col=0)
    # remove duplicates of index
    list_of_ratios_removed_duplicates = []
    list_of_ratios_removed_duplicates.append(listOfRatios[0])
    for i in range(1, len(listOfRatios)):
        if listOfRatios[i] not in listOfRatios[:i]:
            list_of_ratios_removed_duplicates.append(listOfRatios[i])
    print(list_of_ratios_removed_duplicates)
    df_golden_ratios_all = df_ratios.loc[list_of_ratios_removed_duplicates]
    df4ggplot = pd.DataFrame(None)
    df4ggplot['ratio'] = np.nan
    df4ggplot['level'] = np.nan
    df4ggplot['compounds'] = np.nan
    i = 0
    # print(df_golden_ratios_all)
    for index_names in list(df_golden_ratios_all.index.values):
        for column_names in list(df_golden_ratios_all.columns.values):
            if 'RG' in column_names:
                # print(df_golden_ratios_all.loc[index_names, column_names])
                df4ggplot.loc[i, 'ratio'] = df_golden_ratios_all.loc[index_names, column_names]
                df4ggplot.loc[i, 'level'] = 'RG'
                df4ggplot.loc[i, 'compounds'] = index_names
                i += 1
                pass
            elif 'RP' in column_names:
                df4ggplot.loc[i, 'ratio'] = df_golden_ratios_all.loc[index_names, column_names]
                df4ggplot.loc[i, 'level'] = 'RP'
                df4ggplot.loc[i, 'compounds'] = index_names
                i += 1
                pass
            else:
                print('%s is not a valid column'%column_names)
        pass
    df4ggplot.to_excel('../1.data_generated/%s.xlsx'%(df_print_name), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputFeatureImportance', type=str, required=True, help='The Feature Importance File of rfc')
    parser.add_argument('-d', '--dataset', required=True, help='Dataset')
    parser.add_argument('-o', '--output', required=True, help='Output filename for ggplot')
    opt = parser.parse_args()
    df_feature_importance = pd.read_excel(opt.inputFeatureImportance, index_col=0)
    # df_feature_importance = pd.read_excel('../1.data_generated/4.RandomForestTree_featureImportance_2021-12-07.xlsx', index_col=0)
    importance_features_top15 = list(df_feature_importance.index.values)[:15]
    ratios_print = []
    for ratio_name in importance_features_top15:
        ratio_searching = ratio_name.split('-range(')[0]
        ratios_print.append(ratio_searching)
    plot_ratios_violin(file_path=opt.dataset, listOfRatios=ratios_print, df_print_name=opt.output)
    pass



