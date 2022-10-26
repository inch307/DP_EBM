import argparse
import pandas as pd
import csv
import numpy as np
import ebm as ebm
import os
from utils import *
from sklearn import datasets
from dputils import DPUtils
import time

parser = argparse.ArgumentParser(description='DP_EBM')
## 
parser.add_argument('-p', '--data_path', default='data/adult.csv', help='dataset to use: adult, ...')
parser.add_argument('-l', '--label', help='the name of label column')
parser.add_argument('--regression', default=False, action='store_true', help='default is binary classification')
parser.add_argument('--n_runs', type=int, default=1)
parser.add_argument('--seed', type=int, default=2022)

## privacy parameters
parser.add_argument('--privacy', default=False, action='store_true')
parser.add_argument('-r', '--range_label', type=float, help='the range of label for sensitivity')
parser.add_argument('-e', '--eps', type=float, default=0.1, help='epsilon privacy budget of building EBM (0 or negative for non-privacy)')
parser.add_argument('-d', '--delta', type=float, default=1e-6, help='delta for privacy')
parser.add_argument('--hist_ebm_ratio', type=float, default=0.9, help='ebm_eps = eps * hist_ebm_ratio, hist_eps = eps - ebm_eps')
# parser.add_argument('--residual_eps_ratio', type=float, default=0.5, help='residual_eps = ebm_eps * ratio, hessian_eps = ebm_eps - residual_eps ;; only for privacy and classification_hessian')
parser.add_argument('--adaptive_feature', default=False, action='store_true')
parser.add_argument('--af_prob', default=0.01, type=float, help='prune probabilty bound')
parser.add_argument('--re_train', default=False, action='store_true')
parser.add_argument('--ret_epochs', default=30, type=int)
parser.add_argument('--ret_lr', default=0.1, type=float)
parser.add_argument('--max_bins', default=32, type=int)
parser.add_argument('--fake_eps', default=0, type=float)

## tree building
parser.add_argument('--lr', type=float, default=0.01, help='learning rate of EBM')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--max_leaves', type=int, default=3, help='the number of max leaves (max_split +1)')
parser.add_argument('--regularization_score', type=float, default=0, help='regularization term for similarity score')
parser.add_argument('--gamma', type=float, default=0, help='parameter for pruning (gain-gamma)')
# parser.add_argument('outer_bags', type=int, default=0, help='outer bagging')


def main():
    args = parser.parse_args()
    print(args)
    
    if args.data_path == 'syn_class':
        X, y = datasets.make_classification(n_samples=100000, n_features=20, n_informative=10, n_redundant=10, n_clusters_per_class=2, random_state=2022)
        column_name = []
        for i in range(20):
            column_name.append(str(i))
        X_df = pd.DataFrame(X, index=None, columns=column_name)
        y_df = pd.DataFrame(y, index=None, columns=[args.label])
        df =pd.concat([X_df, y_df], axis=1)

    elif args.data_path == 'syn_reg':
        X, y = datasets.make_regression(n_samples=100000, n_features=20, n_informative=10, random_state=2022)
        column_name = []
        for i in range(20):
            column_name.append(str(i))
        X_df = pd.DataFrame(X, index=None, columns=column_name)
        y_df = pd.DataFrame(y, index=None, columns=[args.label])
        df =pd.concat([X_df, y_df], axis=1)
    
    else:
        df = pd.read_csv(args.data_path)
    # print(df)
    if os.path.exists('experiment3.csv'):
        csv_f = open('experiment3.csv', 'a', newline='')
        wr = csv.writer(csv_f)
    else:
        csv_f = open('experiment3.csv', 'w', newline='')
        wr = csv.writer(csv_f)
        write_columns(wr)

    write_lst = make_write_lst(args)

    rmse_lst = []
    acc_lst = []
    auroc_lst = []
    eps_lst = []
    rmse_retrain_lst = []
    acc_retrain_lst = []
    auroc_retrain_lst = []
    for i in range(args.n_runs):
        df_train, df_test = train_test_split(df, 0.8)
        
        model = ebm.EBM(df_train, args)
        model.fit()
        
        train_X = df_train.drop(columns=[args.label], axis=1)
        train_y = df_train[args.label]
        test_X = df_test.drop(columns=[args.label], axis=1)
        test_y = df_test[args.label]

        # predict
        if args.regression:
            rmse = model.predict(test_X, test_y)
            # np.set_printoptions(threshold=np.inf)
            # print(mse)
            rmse_lst.append(rmse)
            
        else:
            accuracy, auroc = model.predict(test_X, test_y)
            # np.set_printoptions(threshold=np.inf)
            # print(y_hat)
            # print(model.intercept)
            # print(accuracy)
            # print(auroc)
            acc_lst.append(accuracy)
            auroc_lst.append(auroc)
        if args.privacy:
            if args.delta == 0:
                remain_eps = model.ebm_eps - model.consumed_eps
                eps_lst.append(args.eps - remain_eps)
            else:
                remain_mu = model.ebm_mu - model.consumed_mu
                if remain_mu < 1e-5:
                    remain_eps = 0
                else:
                    remain_eps = DPUtils.eps_from_mu(remain_mu, args.delta)
                eps_lst.append(args.eps - remain_eps)

        # if re_train
        if args.re_train:
            model.re_train()
            if model.re_trained:
                if args.regression:
                    rmse = model.predict(test_X, test_y)
                    rmse_retrain_lst.append(rmse)
                
                else:
                    accuracy, auroc = model.predict(test_X, test_y)
                    acc_retrain_lst.append(accuracy)
                    auroc_retrain_lst.append(auroc)
            else:
                if args.regression:
                    rmse_retrain_lst.append(rmse)
                
                else:
                    acc_retrain_lst.append(accuracy)
                    auroc_retrain_lst.append(auroc)
        
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

    if args.privacy:
        write_lst.append(np.mean(eps_lst))
        write_lst.append(np.std(eps_lst))
    else:
        write_lst.append('')
        write_lst.append('')
    write_lst.append(args.fake_eps)

    # if args.re_train:
    #     if model.re_trained:
    #         if args.regression:
    #             rmse = np.array(rmse_retrain_lst)
    #             write_lst.append(np.mean(rmse))
    #             write_lst.append(np.std(rmse))
    #             write_lst.append('')
    #             write_lst.append('')
    #             write_lst.append('')
    #             write_lst.append('')
    #         else:
    #             acc = np.array(acc_retrain_lst)
    #             auroc = np.array(auroc_retrain_lst)
    #             write_lst.append('')
    #             write_lst.append('')
    #             write_lst.append(np.mean(acc))
    #             write_lst.append(np.std(acc))
    #             write_lst.append(np.mean(auroc))
    #             write_lst.append(np.std(auroc))
    # else:
    #     write_lst.append('')
    #     write_lst.append('')
    #     write_lst.append('')
    #     write_lst.append('')
    #     write_lst.append('')
    #     write_lst.append('')
        
    wr.writerow(write_lst)
    csv_f.close()

    return

if __name__=='__main__':
    main()