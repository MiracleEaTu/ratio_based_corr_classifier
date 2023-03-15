# This is the code repository work for ratio-based random forest classifier for Baijiu quality evaluation.

## To simply used the classifier, please download this repository to your local disk.
## The following files are required:
    0.rawdata/
        datasheet.xlsx
        odor_threshold.xlsx
        odor_descripter.xlsx

## To run the code, use the following command:
python run_model.py -in [path of the datasheet.xlsx] -ot [path of the odor_threshold.xlsx] -od [path of the odor_descripter.xlsx]

### It may takes anywhere from few hours to few days to run the program based on the size of your dataset.

## Siting information:
Guan, Q., Meng, L. J., Mei, Z., Liu, Q., Chai, L. J., Zhong, X. Z., Zheng, L., Liu, G., Wang, S., Shen, C., Shi, J. S., Xu, Z. H., & Zhang, X. J. (2022). Volatile Compound Abundance Correlations Provide a New Insight into Odor Balances in Sauce-Aroma Baijiu. Foods (Basel, Switzerland), 11(23), 3916. https://doi.org/10.3390/foods11233916
