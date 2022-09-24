import pandas as pd
import random

def category_bin_splits(s):
    x = len(s)
    lst = []
    for i in range(1 << x):
        left = []
        right = []
        for j in range(x):
            if (i & (1 << j)):
                left.append(s[j])
            else:
                right.append(s[j])
        # lst.append([s[j] for j in range(x) if (i & (1 << j))])
        lst.append([left, right])
    return lst[:-1]

def train_test_split(df, train_ratio):
    num_data = len(df)
    slicer = int(num_data * train_ratio)
    shuffled = random.sample(range(num_data), num_data)
    train = df.loc[shuffled[:slicer]]
    test = df.loc[shuffled[slicer:]]

    return train, test

def write_columns(wr):
    wr.writerow(['data', 'n_runs', 'privacy', 'eps', 'delta', 'ss', 'af', 'af_epoch', 'af_prob', 'af_mr', 'af_count', 'cls_lr', 'adaptive_lr', 'lr', 'epo', 'cls_hes', 'rmse', 'rmse_std', 'acc', 'acc_std', 'auroc', 'auroc_std'])

def make_write_lst(args):
    lst = [args.data_path, args.n_runs, args.privacy, args.eps, args.delta, args.split_strategy, args.adaptive_feature, args.af_epoch, args.af_prob, args.af_max_remove, args.af_count, args.cls_lr, args.adaptive_lr, args.lr, args.epochs, args.classification_hessian]

    return lst

if __name__ == '__main__':
    print(powerset([2,3,4]))