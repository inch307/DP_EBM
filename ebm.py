from utils import *
import numpy as np
import math

class EBM():
    def __init__(self, df, args):
        self.df = df
        self.args = args
        if self.args.regression:
            self.regression = True
        else:
            self.regression = False

    def preprocess(self):
        self.data_type = {}

        # process label
        self.label_df = self.df[self.args.label]
        self.df = self.df[self.df.columns.remove(self.args.label)]
        if self.args.regression:
            self.range_label = self.args.range_label
            self.label = self.label_df.to_numpy().astype(float)
        else:
            self.range_label = 1
            self.label = self.label_df.to_numpy().astype(int)

        # if self.args.eps > 0:
        #     if self.args.regression:
        #         if self.args.range_label==None:
        #             self.range_label = self.label_df.max() - self.label_df.min()
        #         else:
        #             self.range_label = self.args.range_label
        #         self.label = self.label_df.to_numpy().astype(float)
        #     else:
        #         self.range_label = 1
        #         self.label = self.label_df.to_numpy().astype(int)


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
                self.histograms[i] = [hist_edges, hist_counts]
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

        #### initializing

        # initialize addtivie terms
        self.intercept = self.args.intercept
        # additiveterms[feature][epoch][additiveterms_per_histogram]
        self.additive_terms = {}
        for i in self.df.columns:
            self.additive_terms[i] = []

        # decision function X: bin in which data fell -> regression or classification
        self.decision_function = {}
        for i in self.df.columns:
            if self.data_type[i] == 0: # numerical
                lst = []
                for j in range(len(self.histograms[i])):
                    lst.append(0)
                self.decision_function[i] = lst
            else:
                self.decision_function[i] = {}
                for j, _ in self.histograms[i]:
                    self.decision_function[i][j] = 0
        
        #initialize split points
        # split_points[feature][epoch][(0,2), (3, 3), (4, 6)] (numerical)
        # split_points[feature][epoch][[0, 3, 1, 2, 5, 6]] (categorical) (split x=0 -> x=3 -> x=1 ...)
        self.split_points = {}
        for i in self.df.columns:
            self.split_points[i] = []
        
        # initialize residuals
        self.residuals = self.label.copy()
        # TODO? compute residuals by intercept
        # access self.residuals[self.hist_idx[feature][bin]]
        
        return

    # numerical split
    def get_split_numerical(self, feature):
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
                    gain = self.get_gain_numerical(feature, split_point, left, right)
                    if max_gain < gain:
                        max_gain, max_split_point, max_left, max_right = gain, split_point, left, right
            split.remove([max_left, max_right])
            split.append([max_left, max_split_point])
            split.append([max_split_point+1, max_right] )

        return split

    def get_gain_numerical(self, feature, split_point, left, right):
        # (left<= x <= split_point), (split_point +1 <= x <= right)
        sim_left = self.get_sim_score_numerical(feature, left, split_point)
        sim_right = self.get_sim_score_numerical(feature, split_point+1, right)
        sim_parent = self.get_sim_score_numerical(feature, left, right)

        return sim_left + sim_right - sim_parent

    def get_sim_score_numerical(self, feature, left, right):
        # if reg or class -> residual
        pass

    # categorical split
    def get_split_categorical(self, feature):
        pass

    def get_gain_categorical(self, feature, parents, split_point):
        sim_left = self.get_sim_score_categorical(feature, parents, split_point, True)
        sim_right = self.get_sim_score_categorical(feature, parents, split_point, False)
        sim_parent = self.get_sim_score_categorical(feature, parents[:-1], parents[-1], False)

        return sim_left + sim_right - sim_parent

    def get_sim_score_categorical(self, feature, parents, split_point, tf):
        # sim_score at split_point with tf under parents

        pass

    def fit(self):
        self.preprocess()
        for epoch in range(self.args.epochs):
            for feature in self.df.columns:
                # get best split
                if self.data_type[feature] == 0: # numerical
                    # left node: idx <= split_point, right node: split_point + 1 <= idx
                    best_splits = self.get_split_numerical(feature)
                    self.split_points[feature].append(best_splits)
                    for split in best_splits:
                        sum_residuals = 0
                        num_hist = 0
                        for i in range(split[0], split[1]+1):
                            sum_residuals += sum(self.residuals[self.hist_idx[feature][i]])
                            num_hist += self.histograms[feature][i][1]
                        avg_residuals = sum_residuals / num_hist

                        for i in range(split[0], split[1]+1):
                            self.decision_function[feature][]



                else: # categorical
                    best_splits = self.get_split_categorical(feature)
                    self.split_points[feature].append(best_splits)
                

            
            #


        return
        
