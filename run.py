import argparse
import pandas as pd
import ebm
import dpebm
from utils import *

parser = argparse.ArgumentParser(description='DP_EBM')
## 
parser.add_argument('-p', '--data_path', default='data/adult.csv', help='dataset to use: adult, ...')
parser.add_argument('-l', '--label', help='the name of label column')
parser.add_argument('--regression', default=False, action='store_true', help='default is binary classification')

## privacy parameters
parser.add_argument('-r', '--range_label', type=float, help='the range of label for sensitivity')
parser.add_argument('-e', '--eps', type=float, default=0.1, help='epsilon privacy budget of building EBM (0 or negative for non-privacy)')
parser.add_argument('-d', '--delta', type=float, default=1e-5, help='delta for privacy')
parser.add_argument('--hist_ebm_ratio', type=float, default=0.9, help='the ratio of privacy budget to histogram and EBM')


## tree building
parser.add_argument('--lr', type=float, default=0.001, help='learning rate of EBM')
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--max_leaves', type=int, default=8, help='the number of max leaves (max_split +1)')
parser.add_argument('--intercept', type=float, default=0, help='intercept of EBM, default is 0')
parser.add_argument('--regularization_score', type=float, default=0, help='regularization term for similarity score')
parser.add_argument('--gamma', type=float, default=0, help='parameter for pruning (gain-gamma)')
# parser.add_argument('outer_bags', type=int, default=0, help='outer bagging')


## histogram building
parser.add_argument('--target_bins', type=int, default=40, help='target bins for differentially private binning')


def main():
    args = parser.parse_args()
    
    df = pd.read_csv(args.data_path)
    model = ebm.EBM(df, args)
    model.fit()

    # collect meta data (generate schema) labels, columns, 


    return

if __name__=='__main__':
    main()