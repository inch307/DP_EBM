from utils import *
import numpy as np
import math

class EBM():
    def __init__(self, df, args):
        self.df = df
        self.args = args
        self.additive_terms = {}
        if self.args.regression:
            self.regression = True
        else:
            self.regression = False

    def preprocess(self):
        self.data_type = {}

        # process label
        self.label_df = self.df[self.args.label]
        self.df = self.df[self.df.columns.remove(self.args.label)]

        if self.args.eps > 0:
            if self.args.regression:
                if self.args.range_label==None:
                    self.range_label = self.label_df.max() - self.label_df.min()
                else:
                    self.range_label = self.args.range_label
                self.label = self.label_df.to_numpy().astype(float)
            else:
                self.range_label = 1
                self.label = self.label_df.to_numpy().astype(int)


        # specify categorical and numerical data type
        # self.data_type (0: numerical data, 1: categorical data)
        for i in self.df.columns:
            if self.df.dtypes[i] == 'object':
                self.data_type[i] = 1
            else:
                self.data_type[i] = 0

        # build historgam
        self.histograms = {}
        self.hist_idx = {}
        for i in self.df.columns:
            col_data = self.df[i].to_numpy()

            # numeric
            if self.data_type[i] == 0:
                hist_counts, hist_edges = np.histogram(col_data, bins="doane")
                self.histograms[i] = [hist_counts, hist_edges]
                self.hist_idx[i] = []
                for j in range(len(hist_counts)):
                    self.hist_idx.append([])
                min = hist_edges[0]
                width = hist_edges[1] - hist_edges[0]
                for idx, val in enumerate(col_data):
                    self.hist_idx[i][math.floor((val - min) / width)].append(idx)

            # categorical
            elif self.data_type[i] == 1:
                uniq_vals, counts = np.unique(col_data, return_counts=True)
                self.histograms[i] = [uniq_vals, counts]
                self.hist_idx[i] = {}
                for j in uniq_vals:
                    self.hist_idx[j] = []
                for idx, val in enumerate(col_data):
                    self.hist_idx[val].append(idx)

        # initialize addtivie terms
        self.intercept = self.args.intercept
        # additiveterms[feature][epoch][additiveterms_per_histogram]
        for i in self.df.columns:
            self.additive_terms[i] = []
        
        # initialize residuals
        self.residuals = []
        # TODO? compute residuals by intercept
        self.residuals.append(self.label)
        
        return

    def get_split(self, feature):
        # split: [(0,1), (2, 4), (5,5), (6, 8)]
        split = [[0, len(self.histograms[feature][0])-1]]
                
        for i in range(self.args.max_leaves-1):
            split.sort(key=lambda x : x[0])
            max_gain = -999999
            max_split_point = None
            max_left = None
            max_right = None
            for left, right in split:
                if left == right:
                    continue
                for p in range(right-left):
                    split_point = left + p
                    gain = self.get_gain(feature, split_point, left, right)
                    if max_gain < gain:
                        max_gain, max_split_point, max_left, max_right = gain, split_point, left, right
            split.remove([max_left, max_right])
            split.append([max_left, max_split_point])
            split.append([max_split_point+1, max_right] )

        return

    def get_sim_score_numerical(self, feature, left, right):
        # if reg or class -> residual
        pass

    def get_sim_score_categorical(self, feature, left, right):
        pass

    def get_gain(self, feature, split_point, left, right):
        # (left<= x <= split_point), (split_point +1 <= x <= right)
        if self.data_type[feature] == 0: # numerical
            sim_left = self.get_sim_score_numerical(feature, left, split_point)
            sim_right = self.get_sim_score_numerical(feature, split_point+1, right)
            sim_parent = self.get_sim_score_numerical(feature, left, right)

        else:
            sim_left = self.get_sim_score_categorical(feature, left, split_point)
            sim_right = self.get_sim_score_categorical(feature, split_point+1, right)
            sim_parent = self.get_sim_score_categorical(feature, left, right)

        return sim_left + sim_right - sim_parent

    def fit(self):
        self.preprocess()
        for epoch in range(self.args.epochs):
            for feature in self.df.columns:
                split_points = []
                for i in self.args.max_leavs-1:
                    # left node: idx < split_point, right node: split_point <= idx
                    split_point = self.get_split(feature)


        return
        
