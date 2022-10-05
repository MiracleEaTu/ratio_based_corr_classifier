import pandas as pd 
import os
import sys

def seperating_descriptors(df_oav, df_descriptor, listOfOdor):
    os.mkdir('../1.data_generated/odor_seperated/')
    dict_odor2index = {}
    for odor_names in listOfOdor:
        dict_odor2index[odor_names] = []
    print(dict_odor2index)
    for index_values in df_oav.index.values:
        description_of_index = df_descriptor.loc[index_values, 'Descriptor']
        if type(description_of_index) == str:
            description_of_index_lower = description_of_index.lower()
        else:
            description_of_index_lower = ''
        for odor_names in listOfOdor:
            if odor_names in description_of_index_lower:
                raw_list = dict_odor2index[odor_names]
                print(raw_list, odor_names, index_values)
                raw_list.append('%s'%index_values)
                dict_odor2index[odor_names] = raw_list
                # print(raw_list.append(index_values))
                print(dict_odor2index[odor_names])
            pass
    for odor_names in listOfOdor:
        df_out = df_oav.loc[dict_odor2index[odor_names]]
        print('Number of %s-like odors after filteration is %i'%(odor_names, len(df_out)))
        if len(df_out) >= 2:
            df_out.to_excel('../1.data_generated/odor_seperated/oav_%s.xlsx'%odor_names, index=True)
        del(df_out)
    return 0



if __name__ == '__main__':
    df_oav = pd.read_excel('../1.data_generated/2.data_transfered2oav.xlsx', index_col=0)
    df_descriptor = pd.read_excel(sys.argv[1], index_col=0)
    list_of_odors = ['fruity', 'alcoholic', 'sweaty', 'sweet', 'fatty', 'nutty', 'malt', \
        'plant', 'cheese', 'floral', 'vinegar']
    seperating_descriptors(df_oav=df_oav, df_descriptor=df_descriptor, listOfOdor=list_of_odors)
