import numpy as np 
import pandas as pd 
import collections
import os
import sys
import datetime
__author__ = 'Qijie Guan'

# data set filtration 
def data_cleaning(df_in_path, level_good_name, level_poor_name):
    df_in = pd.read_excel(df_in_path, index_col=0)
    y = []
    colums_using = [names for names in df_in.columns.values if '%s'%level_good_name in names or '%s'%level_poor_name in names]
    df_using = df_in[colums_using]
    # 第一次数据清洗：删除1/4以上未检出的数据，进行ratio计算
    df_filtered_by_nonzeros = pd_remove_rows_by_nonzeros(df_using)
    df_filtered_by_nonzeros.to_excel('../1.data_generated/0.datasheet_filtered_by_nonzeros.xlsx', index=True)
    df_extended = generate_ratios_logged(df_filtered_by_nonzeros)
    # df_extended.to_excel('../1.data_generated/1.df_extended_from_nonzerofiltered_%s.xlsx'%str(datetime.date.today()), index=True)
    # 根据各分组来确定std
    df_level_good = df_extended[[names for names in df_extended.columns.values if '%s'%level_good_name in names]]
    df_level_poor = df_extended[[names for names in df_extended.columns.values if '%s'%level_poor_name in names]]
    df_level_0_using = pd_remove_rows_by_std(df_level_good)
    df_level_1_using = pd_remove_rows_by_std(df_level_poor)
    # 留下至少在两组中std符合要求的数据集
    index_all_remaining = list(df_level_0_using.index.values) + list(df_level_1_using.index.values) 
    keys_all = dict(collections.Counter(index_all_remaining))
    index_replicates = [key for key,value in keys_all.items() if value > 1]
    # print(len(index_replicates))
    df_filtered_by_std_and_nonzeros = df_extended.loc[index_replicates]
    return df_filtered_by_std_and_nonzeros



def pd_remove_rows_by_std(df_in):
    print('Start deleting rows because of high row variation...')
    index_remain = []
    for index_name in df_in.index.values:
        row_data = df_in.loc[index_name]
        # 如果该行数据标准差小于均值2/3则保留该行。
        if np.std(row_data)*3 <= np.average(row_data)*2:
            index_remain.append(index_name)
    df_remaining = df_in.loc[index_remain]
    print('Before data cleaning, %i rows were used. \n\tAfter filteration with std, %i rows were remained'%(len(df_in.index.values), len(index_remain)))
    return df_remaining


def pd_remove_rows_by_nonzeros(df_in):
    print('Start deleting rows based on too many zero values...')
    index_remain = []
    for index_name in df_in.index.values:
        row_data = df_in.loc[index_name]
        # 未检出数据量小于该行数据的1/4，则保留该行。
        if np.count_nonzero(row_data)*4 >= len(row_data)*3:
            index_remain.append(index_name)
    df_remaining = df_in.loc[index_remain]
    print('Before data cleaning, %i rows were used. \n\tAfter filteration with nonzeros, %i rows were remained'%(len(df_in.index.values), len(index_remain)))
    return df_remaining


# 将两两比值数据加到数据集中
def generate_ratios_logged(df_in):
    print('Start ratio generation...')
    df_extended = pd.DataFrame(None, columns=df_in.columns.values)
    n = len(df_in.index.values)
    for i in range(n-1):
        for j in range(i+1, n):
            df_extended.loc['%s/%s'%(df_in.index.values[i],df_in.index.values[j])] = np.log2(np.divide(df_in.loc[df_in.index.values[i]]+0.01, df_in.loc[df_in.index.values[j]]+0.01))
    print('\tFinished logged ratios generation')
    return df_extended


def main(df_raw_path='../0.rawdata/datasheet.xlsx'):
    # if data_output folder not exist, create the folder.
    path_all = os.listdir('../')
    if '1.data_generated' not in path_all:
        os.mkdir('../1.data_generated/')
    df_cleaned = data_cleaning(df_raw_path, level_good_name='RG', level_poor_name='RP')
    df_cleaned.to_excel('../1.data_generated/1.ratio_datasheet_filtered_by_nonzero_and_std.xlsx', index=True)


if __name__ == '__main__':
    main(df_raw_path=sys.argv[1])