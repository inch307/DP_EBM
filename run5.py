import argparse
import pandas as pd
import csv
import numpy as np
import ebm
import dpebm
import os
from utils import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='DP_EBM')
## 
parser.add_argument('-p', '--data_path', default='data/adult.csv', help='dataset to use: adult, ...')
parser.add_argument('-l', '--label', help='the name of label column')
parser.add_argument('--regression', default=False, action='store_true', help='default is binary classification')
parser.add_argument('--classification_hessian', default=False, action='store_true')
parser.add_argument('--n_runs', type=int, default=1)
parser.add_argument('--seed', type=int, default=2022)

## privacy parameters
parser.add_argument('--privacy', default=False, action='store_true')
parser.add_argument('-r', '--range_label', type=float, help='the range of label for sensitivity')
parser.add_argument('-e', '--eps', type=float, default=0.1, help='epsilon privacy budget of building EBM (0 or negative for non-privacy)')
parser.add_argument('-d', '--delta', type=float, default=1e-6, help='delta for privacy')
parser.add_argument('--hist_ebm_ratio', type=float, default=0.9, help='ebm_eps = eps * hist_ebm_ratio, hist_eps = eps - ebm_eps')
# parser.add_argument('--residual_eps_ratio', type=float, default=0.5, help='residual_eps = ebm_eps * ratio, hessian_eps = ebm_eps - residual_eps ;; only for privacy and classification_hessian')
parser.add_argument('--split_strategy', default=False, action='store_true', help='False: random split, True: split_strategy')
parser.add_argument('--adaptive_feature', default=False, action='store_true')
parser.add_argument('--adaptive_lr', default=False, action='store_true', help='adpative lr when adaptive feature')
parser.add_argument('--af_epoch', default=1, type=int, help='prune feature each af_epoch')
parser.add_argument('--af_prob', default=0.01, type=float, help='prune probabilty bound')
parser.add_argument('--af_max_remove', default=2, type=int, help='at each af_epoch, #af_max_prune features can be removed at most')
parser.add_argument('--af_threshold_column', default=False, help='false: analytic adaptive_feature')
parser.add_argument('--af_count', default=1, type=int)
parser.add_argument('--max_bins', default=32, type=int)





## tree building
parser.add_argument('--lr', type=float, default=0.001, help='learning rate of EBM')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--max_leaves', type=int, default=3, help='the number of max leaves (max_split +1)')
parser.add_argument('--regularization_score', type=float, default=0, help='regularization term for similarity score')
parser.add_argument('--gamma', type=float, default=0, help='parameter for pruning (gain-gamma)')
# parser.add_argument('outer_bags', type=int, default=0, help='outer bagging')


def main():
    args = parser.parse_args()
    print(args)
    
    df = pd.read_csv(args.data_path)
    # print(df)
    if os.path.exists('experiment5.csv'):
        csv_f = open('experiment5.csv', 'a', newline='')
        wr = csv.writer(csv_f)
    else:
        csv_f = open('experiment5.csv', 'w', newline='')
        wr = csv.writer(csv_f)
        write_columns(wr)

    write_lst = make_write_lst(args)

    rmse_lst = []
    acc_lst = []
    auroc_lst = []
    for i in range(args.n_runs):
        df_train, df_test = train_test_split(df, 0.8)
        
        model = ebm.EBM(df_train, args)
        model.fit()
        
        train_X = df_train.drop(columns=[args.label], axis=1)
        train_y = df_train[args.label]
        test_X = df_test.drop(columns=[args.label], axis=1)
        test_y = df_test[args.label]
        if args.regression:
            rmse = model.predict(test_X, test_y)
            np.set_printoptions(threshold=np.inf)
            # print(mse)
            rmse_lst.append(rmse)
            
        else:
            accuracy, auroc = model.predict(test_X, test_y)
            np.set_printoptions(threshold=np.inf)
            # print(y_hat)
            # print(model.intercept)
            # print(accuracy)
            # print(auroc)
            acc_lst.append(accuracy)
            auroc_lst.append(auroc)
            

        # collect meta data (generate schema) labels, columns, 
    if args.regression:
        rmse = np.array(rmse_lst)
        write_lst.append(np.mean(rmse))
        write_lst.append(np.std(rmse))
        write_lst.append('')
        write_lst.append('')
        write_lst.append('')
        write_lst.append('')
    else:
        acc = np.array(acc_lst)
        auroc = np.array(auroc_lst)
        write_lst.append('')
        write_lst.append('')
        write_lst.append(np.mean(acc))
        write_lst.append(np.std(acc))
        write_lst.append(np.mean(auroc))
        write_lst.append(np.std(auroc))
    wr.writerow(write_lst)
    csv_f.close()

    return

if __name__=='__main__':
    main()