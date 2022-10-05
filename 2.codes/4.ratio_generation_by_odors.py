from numpy.core.defchararray import index
from numpy.matrixlib.defmatrix import matrix
import pandas as pd 
import numpy as np 
__author__ = 'Qijie Guan'

# Ratio generate by: log2(Compound_1 / Compound_2) 
def generate_ratios(odor_des_list, df_out_path):
    df_path_list = []
    for names in odor_des_list:
        df_path_list.append('../1.data_generated/odor_seperated/oav_%s.xlsx'%names)
    df_extended = pd.DataFrame(None, columns=pd.read_excel(df_path_list[0], index_col=0).columns.values)
    for n in range(len(df_path_list)-1):
        for m in range(n+1, len(df_path_list)):
            df_in_1_raw = pd.read_excel(df_path_list[n], index_col=0)
            df_in_1 = pd_remove_rows_by_nonzeros(df_in_1_raw)
            df_in_2_raw = pd.read_excel(df_path_list[m], index_col=0)
            df_in_2 = pd_remove_rows_by_nonzeros(df_in_2_raw)
            # n = len(df_in.index.values)
            df_extended.loc['%s/%s'%(odor_des_list[n], odor_des_list[m])] = np.log2(np.divide(np.sum(df_in_1)+0.01, np.sum(df_in_2)+0.01))
            for i in range(len(df_in_1.index.values)):
                for j in range(len(df_in_2.index.values)):
                    # print(i,j)
                    df_extended.loc['%s/%s'%(df_in_1.index.values[i],df_in_2.index.values[j])] = np.log2(np.divide(df_in_1.loc[df_in_1.index.values[i]]+0.01, df_in_2.loc[df_in_2.index.values[j]]+0.01))
            print('\tLogged ratio generation of %s / %s is finished!'%(df_path_list[n], df_path_list[m]))
    print('Finished ratios generation\n\tFile saving to %s'%df_out_path)
    df_extended.to_excel(df_out_path, index=True)
    return df_extended

def pd_remove_rows_by_nonzeros(df_in):
    print('Start deleting rows based on too many zero values...')
    index_remain = []
    for index_name in df_in.index.values:
        row_data = df_in.loc[index_name]
        # 未检出数据量小于该行数据的1/2，则保留该行。
        if np.count_nonzero(row_data)*2 >= len(row_data):
            index_remain.append(index_name)
    df_remaining = df_in.loc[index_remain]
    print('Before data cleaning, %i rows were used. \n\tAfter filteration with nonzeros, %i rows were remained'%(len(df_in.index.values), len(index_remain)))
    return df_remaining



def generate_ratios_main():
    odor_des_list = ['fruity', 'alcoholic', 'sweaty', 'sweet', 'nutty', 'malt', 'floral']
    df_log2ratios = generate_ratios(odor_des_list=odor_des_list, df_out_path='../1.data_generated/4.log2ratios_generated_by_oavs.xlsx')
    # df_oav = pd.read_excel('../1.data_generated/02.oav_data_removed_outliers_20211221.xlsx', index_col=0)
    # df_log2ratio_w_oav = pd.concat([df_log2ratios, df_oav])
    # df_log2ratio_w_oav.to_excel('../1.data_generated/04.log2ratios_w_oavs_20211221.xlsx', index=True)



if __name__ == '__main__':
    generate_ratios_main()


