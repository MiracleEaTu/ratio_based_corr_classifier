import argparse
import os
__author__ = 'Qijie Guan'


parse = argparse.ArgumentParser(prog = 'argparse')
parse.add_argument('-in', '--datasheet', required=True, help='The path of compound quantitative file.')
parse.add_argument('-ot', '--odorthreshold', required=True, help='The path of Odor threshold table.')
parse.add_argument('-od', '--odordescriptor', required=True, help='The path of Odor descriptor table for each compound inported')
args = parse.parse_args()  


os.system('python 0.data_cleaning.py %s'%args.datasheet)
os.system('python 1.data_transform2oav.py %s'%args.odorthreshold)
os.system('python 2.odor_seperation.py %s'%args.odordescriptor)
os.system('python 3.PCA_w_remove_outliers.py')
os.system('python 4.ratio_generation_by_odors.py')
os.system('python 5.randomForest_oav_only.py')
os.system('python 6.radomForest_ratios_only.py')
os.system('python 7.randomForest_oav_w_ratio.py')
os.system('python 8.make_boxplot_files4R.py')
