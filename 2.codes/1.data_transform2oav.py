import pandas as pd 
import numpy as np
import sys
from calculations4dataset import dataset_cleaning 
__author__ = 'Qijie Guan'


# 0. change compound quantities to OAVs
def calculate_OAVs4all(df_in, path_odor_thershold='../0.rawdata/odor_thresholds.xlsx'):
    df_OAV = pd.read_excel(path_odor_thershold, index_col=0)
    df_out = pd.DataFrame(None, index=df_in.index.values, columns=df_in.columns.values)
    for index_names in df_in.index.values:
        oav4index = df_OAV.loc[index_names, 'odor threshold']
        if oav4index != 0:
            df_out.loc[index_names] = np.divide(df_in.loc[index_names], oav4index)
        else:
            df_out.loc[index_names] = np.nan
    df_out_cleaned = dataset_cleaning(df_out)
    df_out_cleaned.to_excel('../1.data_generated/2.data_transfered2oav.xlsx', index=True)
    return 0


## main
if __name__ == '__main__':
    df_renamed = pd.read_excel('../1.data_generated/0.datasheet_filtered_by_nonzeros.xlsx', index_col=0)
    calculate_OAVs4all(df_in=df_renamed, path_odor_thershold=sys.argv[1])