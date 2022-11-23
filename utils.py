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
    # return train, test idx list (not df)
    num_data = len(df)
    slicer = int(num_data * train_ratio)
    shuffled = random.sample(range(num_data), num_data)
    # train = df.loc[shuffled[:slicer]]
    # test = df.loc[shuffled[slicer:]]

    return shuffled[:slicer], shuffled[slicer:]

class Dataset:
    def __init__(self, data_path, )

class CrossValidation:
    def __init__(self, df, k_fold, train_ratio):
        self.df = df
        self.k_fold = k_fold
        self.shuffled = random.sample(range(len(self.df)), len(self.df))
        self.slicer = int(len(self.df) / self.k_fold)
        self.i = 0
    
    def get_train_test(self):
        train_left = self.df.loc[self.shuffled[:self.slicer * self.i]]
        train_right = self.df.loc[self.shuffled[min(self.slicer * (self.i+1), len(self.df)):]]
        train = pd.concat([train_left, train_right])
        test = self.df.loc[self.shuffled[self.slicer * self.i: min(self.slicer * (self.i+1), len(self.df))]]
        self.i += 1

        return train, test

def write_columns(wr):
    # wr.writerow(['data', 'n_runs', 'privacy', 'eps', 'delta', 'af', 'af_prob', 'lr', 'epo', 're_train', 'rmse', 'rmse_std', 'acc', 'acc_std', 'auroc', 'auroc_std', 'remain_eps', 're_rmse', 'std', 're_acc', 'std', 're_roc', 'std'])
    wr.writerow(['data', 'n_runs', 'cv', 'privacy', 'eps', 'delta', 'af', 'af_prob', 'min_cf', 'lr', 'epo', 'rmse', 'rmse_std', 'acc', 'acc_std', 'auroc', 'auroc_std'])

def make_write_lst(args):
    lst = [args.data_path, args.n_runs, args.cv, args.privacy, args.eps, args.delta, args.adaptive_feature, args.af_prob, args.min_cf, args.lr, args.epochs]

    return lst

if __name__ == '__main__':
    print(powerset([2,3,4]))