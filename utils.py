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
    # wr.writerow(['data', 'n_runs', 'privacy', 'eps', 'delta', 'af', 'af_prob', 'lr', 'epo', 're_train', 'rmse', 'rmse_std', 'acc', 'acc_std', 'auroc', 'auroc_std', 'remain_eps', 're_rmse', 'std', 're_acc', 'std', 're_roc', 'std'])
    wr.writerow(['data', 'n_runs', 'privacy', 'eps', 'delta', 'af', 'af_prob', 'lr', 'epo', 're_train', 'rmse', 'rmse_std', 'acc', 'acc_std', 'auroc', 'auroc_std', 'actual_eps', 'actual_eps_std' , 'fake_eps'])

def make_write_lst(args):
    lst = [args.data_path, args.n_runs, args.privacy, args.eps, args.delta, args.adaptive_feature, args.af_prob, args.lr, args.epochs, args.re_train]

    return lst

if __name__ == '__main__':
    print(powerset([2,3,4]))