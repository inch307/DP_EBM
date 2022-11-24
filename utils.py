import pandas as pd
import random
from sklearn import datasets

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

def train_test_idx(df, train_ratio):
    # return train, test idx list (not df)
    num_data = len(df)
    slicer = int(num_data * train_ratio)
    shuffled = random.sample(range(num_data), num_data)
    # train = df.loc[shuffled[:slicer]]
    # test = df.loc[shuffled[slicer:]]

    return shuffled[:slicer], shuffled[slicer:]

def get_train_test_df(df, train_idx, test_idx):
    train = df.loc[train_idx]
    test = df.loc[test_idx]

    return train, test

def get_dataset(args):
    if args.data_path == 'syn_cls':
        data_name = 'syn_cls'
        n_features = 60
        X, y = datasets.make_classification(n_samples=10000, n_features=n_features, n_informative=5, n_redundant=5, n_clusters_per_class=2, random_state=args.seed)
        column_name = []
        for i in range(n_features):
            column_name.append(str(i))
        X_df = pd.DataFrame(X, index=None, columns=column_name)
        y_df = pd.DataFrame(y, index=None, columns=[args.label])
        df =pd.concat([X_df, y_df], axis=1)

    elif args.data_path == 'syn_reg':
        data_name= 'syn_reg'
        n_features = 60
        X, y = datasets.make_regression(n_samples=10000, n_features=n_features, n_informative=10, random_state=args.seed)
        column_name = []
        for i in range(n_features):
            column_name.append(str(i))
        X_df = pd.DataFrame(X, index=None, columns=column_name)
        y_df = pd.DataFrame(y, index=None, columns=[args.label])
        df =pd.concat([X_df, y_df], axis=1)
    
    else:
        data_name = args.data_path[5:].split('.')[0]
        df = pd.read_csv(args.data_path)

    return df, data_name

class CrossValidation:
    def __init__(self, df, train_idx, k_fold):
        self.df = df
        self.train_idx = train_idx
        self.k_fold = k_fold
        self.slicer = int(len(self.train_idx) / self.k_fold)
        self.i = 0
    
    def get_train_test(self):
        train_left = self.df.loc[self.train_idx[:self.slicer * self.i]]
        train_right = self.df.loc[self.train_idx[min(self.slicer * (self.i+1), len(self.df)):]]
        train = pd.concat([train_left, train_right])
        test = self.df.loc[self.train_idx[self.slicer * self.i: min(self.slicer * (self.i+1), len(self.df))]]
        self.i += 1

        return train, test

def write_columns(wr):
    # wr.writerow(['data', 'n_runs', 'privacy', 'eps', 'delta', 'af', 'af_prob', 'lr', 'epo', 're_train', 'rmse', 'rmse_std', 'acc', 'acc_std', 'auroc', 'auroc_std', 'remain_eps', 're_rmse', 'std', 're_acc', 'std', 're_roc', 'std'])
    wr.writerow(['data', 'n_runs', 'cv', 'seed', 'privacy', 'eps', 'delta', 'af', 'af_prob', 'min_cf', 'lr', 'epo', 'rmse', 'rmse_std', 'acc', 'acc_std', 'auroc', 'auroc_std'])

def make_write_lst(args):
    lst = [args.data_path, args.n_runs, args.cv, args.seed, args.privacy, args.eps, args.delta, args.adaptive_feature, args.af_prob, args.min_cf, args.lr, args.epochs]

    return lst

if __name__ == '__main__':
    print(powerset([2,3,4]))