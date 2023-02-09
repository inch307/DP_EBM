from utils import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import numpy as np
import math
import random
import scipy
import warnings
import matplotlib.pyplot as plt

from dputils import DPUtils

NUMERICAL = 0
CATEGORICAL = 1
warnings.filterwarnings('ignore', message='overflow encountered in exp')

class EBM():
    def __init__(self, df, args):
        self.df = df
        self.args = args
        self.lr = self.args.lr
        if self.args.delta == 0:
            self.ebm_eps = self.args.eps * self.args.hist_ebm_ratio
            self.hist_eps = self.args.eps - self.ebm_eps
        else:
            self.total_mu_2 = DPUtils.calc_gdp_mu(self.args.eps, delta=self.args.delta)**2
            self.ebm_mu_2 = self.args.hist_ebm_ratio * self.total_mu_2
            self.hist_mu_2 = self.total_mu_2 - self.ebm_mu_2
            

    def preprocess(self):
        self.data_type = {}
        # print('Initializing...')
        # print('Constructing histograms...')

        # process label
        self.label_df = self.df[self.args.label]
        # privacy range label
        if self.args.privacy:
            if self.args.regression:
                if self.args.range_label==None:
                    self.range_label = self.label_df.max() - self.label_df.min()
                else:
                    self.range_label = self.args.range_label
            else:
                self.range_label = 1.0
        self.df = self.df.drop(columns=[self.args.label], axis=1)
        self.label = self.label_df.to_numpy().astype(float)

        # specify categorical and numerical data type
        # self.data_type (0: numerical data, 1: categorical data)
        for i in self.df.columns:
            if self.df.dtypes[i] == 'object':
                self.data_type[i] = CATEGORICAL
            else:
                self.data_type[i] = NUMERICAL
        # print(self.data_type)

        # build historgam
        #TODO: nan data
        self.histograms = {}
        self.hist_idx = {}
        # columns to drop
        self.columns_drop = []
        self.hist_columns = []

        # differentially private histogram building
        if self.args.privacy:
            self.feature_DPOthers = {}
            if self.args.delta == 0:
                self.hist_noise_scale = len(self.df.columns) / self.hist_eps
                lap = True
            else:
                self.hist_noise_scale = DPUtils.noise_from_mu(len(self.df.columns), np.sqrt(self.hist_mu_2))
                lap = False
            for i in self.df.columns:
                col_data = self.df[i].to_numpy()
                if self.data_type[i] == NUMERICAL:
                    min_val = np.nanmin(col_data)
                    max_val = np.nanmax(col_data)
                    hist_edges, hist_counts = DPUtils.private_numeric_binning(col_data=col_data,sample_weight=None, noise_scale=self.hist_noise_scale, max_bins=self.args.max_bins, min_val=min_val, max_val=max_val, lap=lap)
                    if len(hist_counts) == 0:
                        self.columns_drop.append(i)
                        continue
                    self.hist_columns.append(i)
                    self.histograms[i] = {'bin':hist_edges[:-1], 'count':hist_counts}
                    self.hist_idx[i] = []
                    for j in range(len(hist_counts)):
                        self.hist_idx[i].append([])
                    for idx, val in enumerate(col_data):
                        prev_left = 0
                        for j in range(len(hist_counts)):
                            if hist_edges[j] <= val:
                                prev_left = j
                                continue
                            else:
                                break
                        self.hist_idx[i][prev_left].append(idx)
                else:
                    # 'DPOther' contains other uniq vals
                    uniq_vals, bin, counts = DPUtils.private_categorical_binning(col_data=col_data, sample_weight=None, noise_scale=self.hist_noise_scale, max_bins=self.args.max_bins, lap=lap)
                    count_dict = {}
                    self.hist_columns.append(i)
                    for b, c in zip(bin, counts):
                        count_dict[b] = c
                    # transform masked uniq vals to DPOther
                    self.feature_DPOthers[i] = []
                    if bin[-1] == 'DPOther':
                        for b in uniq_vals:
                            if b not in bin:
                                self.feature_DPOthers[i].append(b)
                        self.df[i] = self.df[i].replace(self.feature_DPOthers[i], 'DPOther')
                    
                    self.histograms[i] = {'bin':bin, 'count':count_dict}
                    self.hist_idx[i] = {}
                    for j in bin:
                        self.hist_idx[i][j] = []
                    for idx, col in enumerate(col_data):
                        self.hist_idx[i][col].append(idx)

        # Non private histogram building
        else:
            for i in self.df.columns:
                col_data = self.df[i].to_numpy()

                # numeric
                if self.data_type[i] == NUMERICAL:
                    hist_counts, hist_edges = np.histogram(col_data, bins="doane")
                    if len(hist_edges) == 0:
                        self.columns_drop.append(i)
                        continue
                    self.histograms[i] = {'bin':hist_edges[:-1], 'count':hist_counts}
                    self.hist_idx[i] = []
                    for j in range(len(hist_counts)):
                        self.hist_idx[i].append([])
                    for idx, val in enumerate(col_data):
                        prev_left = 0
                        # TODO: hist edges 0?
                        for j in range(len(hist_counts)):
                            if hist_edges[j] <= val:
                                prev_left = j
                                continue
                            else:
                                break
                        self.hist_idx[i][prev_left].append(idx)

                # categorical
                elif self.data_type[i] == CATEGORICAL:
                    uniq_vals, counts = np.unique(col_data, return_counts=True)
                    count_dict = {}
                    for b, c in zip(uniq_vals, counts):
                        count_dict[b] = c
                    self.histograms[i] = {'bin':uniq_vals, 'count':count_dict}
                    self.hist_idx[i] = {}
                    for j in uniq_vals:
                        self.hist_idx[i][j] = []
                    for idx, col in enumerate(col_data):
                        self.hist_idx[i][col].append(idx)

        # print('histogram done')
        #### initializing
        # total data
        if self.args.privacy:
            total_data_sum = 0.
            total_data_nom = 0.
            for i in self.hist_columns:
                if self.data_type[i] == NUMERICAL:
                    for j in self.histograms[i]['count']:
                        total_data_sum += j
                else:
                    for j in self.histograms[i]['count'].values():
                        total_data_sum += j
                total_data_nom += 1.
            
            self.total_data = int(total_data_sum // total_data_nom)
        else:
            self.hist_columns = self.df.columns.tolist()
            self.total_data = len(self.df[self.hist_columns[0]])

        # initialize addtivie terms
        self.intercept = 0.
        # additiveterms[epoch][feature] = {'split': [[0], [1, 2], [3, 4, 5], [6]], 'additive_term': [3, 5, 8, 9]]}
        self.additive_terms = []

        # decision function X: bin in which data fell -> regression or classification
        self.decision_function = {}
        for i in self.hist_columns:
            if self.data_type[i] == NUMERICAL:
                lst = []
                for j in range(len(self.histograms[i]['count'])):
                    lst.append(0.)
                self.decision_function[i] = lst
            else:
                self.decision_function[i] = {}
                for j in self.histograms[i]['bin']:
                    self.decision_function[i][j] = 0.
        
        # initialize residuals
        self.residuals = self.label.copy()
        if not self.args.regression:
            self.residuals = self.residuals - 0.5
        
        return

    def get_af_threshold(self, feature, epoch, num_data_split):
        def regularized_gamma(x):
            return scipy.special.gammainc(k, x/theta) - self.args.af_prob
        split = self.additive_terms[epoch][feature]['split']
        #  Welch–Satterthwaite equation 
    
        
        if self.args.delta == 0:
            # sum of exponential distribution (1, 1/lambda)
            k = len(split)
            theta = self.residual_noise_scale * self.range_label
        else:
            # Gaussian DP
            # # sum of Gamma(1/2, 2*sigma^2) -> Gamma(n/2, 2*sigma^2)
            # k = len(split) / 2
            # theta = 2 * ((self.range_label * self.residual_noise_scale) ** 2) / self.total_data
            sum_theta_k = 0.
            sum_thetasq_k = 0.
            sigma = (self.range_label * self.residual_noise_scale)
            tt1 = 0
            tt2 = 0
            # print(f'nds: {num_data_split}')
            for num_data in num_data_split:
                sum_theta_k += (sigma**2 / num_data )
                tt1 += 1/num_data
                tt2 += 2 / ((num_data)**2)
                sum_thetasq_k += (2 * sigma**4) / (num_data**2)
            k = (sum_theta_k)**2 / sum_thetasq_k
            # print('kkkk')
            # print(k)
            # print((tt1)**2 / (tt2))
            theta = sum_theta_k / k
            # print('thetatheta')
            # print(theta)
            # print((tt2*self.range_label**2*self.residual_noise_scale**2) / (tt1))
            
  
        sol = scipy.optimize.root_scalar(regularized_gamma,bracket=[1e-16, k*theta/(1-self.args.af_prob)],method='brentq')
        # print(f'sol:{sol}')
        return sol.root

    def get_histogram_residual(self, feature):
        num_bins = len(self.histograms[feature]['count'])
        if self.data_type[feature] == NUMERICAL:
            residuals = [0. for i in range(num_bins)]
            for b in range(num_bins):
                for idx in self.hist_idx[feature][b]:
                    residuals[b] += self.residuals[idx]
        else:
            residuals = {}
            for c in self.histograms[feature]['bin']:
                residuals[c] = 0.
                for idx in self.hist_idx[feature][c]:
                    residuals[c] += self.residuals[idx]
        
        return residuals

    def get_histogram_hessian(self, feature):
        # hessian for classification
        # if (not self.args.privacy):
        #     num_bins = len(self.histograms[feature]['count'])
        #     if self.data_type[feature] == NUMERICAL:
        #         hessian = [0 for i in range(num_bins)]
        #         for b in range(num_bins):
        #             for idx in self.hist_idx[feature][b]:
        #                 hessian[b] += abs(self.residuals[idx]) * (1- abs(self.residuals[idx]))
        #     else:
        #         hessian = {}
        #         for c in self.histograms[feature]['bin']:
        #             hessian[c] = 0
        #             for idx in self.hist_idx[feature][c]:
        #                 hessian[c] += abs(self.residuals[idx]) * (1- abs(self.residuals[idx]))
        #     return hessian
        # else:
        return self.histograms[feature]['count']
        

    # numerical split
    def get_split_numerical(self, feature, histogram_residuals, histogram_hessian):
        # split: [[0,1], [2, 3, 4], [5], [6, 7, 8]]

        # random split
        if self.args.privacy:
            num_bins = len(self.histograms[feature]['count'])
            if num_bins < self.args.max_leaves:
                split = []
                split_points = [0]
                for i in range(num_bins-1):
                    split_points.append(1)
                lst = []
                for idx, v in enumerate(split_points):
                    if v == 0:
                        lst.append(idx)
                    else:
                        split.append(lst)
                        lst = [idx]
                split.append(lst)
                # print(f'sp: {split_points}')
                # print(f's: {split}')
            
            else:
                split = []
                split_points = [0 for i in range(num_bins)]
                points = random.sample(range(1, num_bins), k=self.args.max_leaves-1)
                for i in points:
                    split_points[i] = 1
                lst = []
                for idx, v in enumerate(split_points):
                    if v == 0:
                        lst.append(idx)
                    else:
                        split.append(lst)
                        lst = [idx]
                split.append(lst)
            # print(f'sp: {split_points}')
            # print(f's: {split}')

        # sub-optimal split
        else: 
            split = [[i for i in range(len(self.histograms[feature]['count']))]]
            for i in range(self.args.max_leaves-1):
                max_gain = 0
                max_split = split.copy()
                for idx, parent in enumerate(split):
                    sim_parent = self.get_sim_score(parent, histogram_residuals, histogram_hessian)
                    # if len(parent) == 1 no more split
                    for j in range(len(parent)-1):
                        left_split = parent[0:j+1]
                        right_split = parent[j+1:]
                        # sim = [[num_res, sum_res]]
                        # get_sim_score_numerical(left_split, sim) => get sim_score from sim list
                        sim_left = self.get_sim_score(left_split, histogram_residuals, histogram_hessian)
                        sim_right = self.get_sim_score(right_split, histogram_residuals, histogram_hessian)
                        gain = sim_left + sim_right - sim_parent
                        if max_gain < gain:
                            max_gain = gain
                            copied_split = split.copy()
                            del copied_split[idx]
                            copied_split.insert(idx, left_split)
                            copied_split.insert(idx+1, right_split)
                            max_split = copied_split
                if max_gain == 0:
                    break
                split = max_split.copy()
        return split

    def get_sim_score(self, bins, histogram_residuals, histogram_hessian):
        sum_hessian = 0.
        sum_res = 0.
        for bin in bins:
            sum_hessian += histogram_hessian[bin]
            sum_res += histogram_residuals[bin]
        if sum_hessian <= 0.:
            return 0.
        return sum_res**2 / (sum_hessian + self.args.regularization_score)

    # categorical split
    def get_split_categorical(self, feature, histogram_residuals, histogram_hessian):
        # random split
        if self.args.privacy:
            num_bins = len(self.histograms[feature]['count'])
            if num_bins < self.args.max_leaves:            
                bins = self.histograms[feature]['bin'].tolist()
                random.shuffle(bins)
                
                split = []
                split_points = [0]
                for i in range(num_bins-1):
                    split_points.append(1)
                lst = []
                for idx, v in enumerate(split_points):
                    if v == 0:
                        lst.append(bins[idx])
                    else:
                        split.append(lst)
                        lst = [bins[idx]]
                split.append(lst)

            else:
                bins = self.histograms[feature]['bin'].tolist()
                random.shuffle(bins)
                split = []
                split_points = [0 for i in range(num_bins)]
                points = random.sample(range(1, num_bins), k=self.args.max_leaves-1)
                for i in points:
                    split_points[i] = 1
                lst = []
                for idx, v in enumerate(split_points):
                    if v == 0:
                        lst.append(bins[idx])
                    else:
                        split.append(lst)
                        lst = [bins[idx]]
                split.append(lst)
                # print(f'sp: {split_points}')
                # print(f's: {split}')

        # sub-optimal split
        else:
            bins = self.histograms[feature]['bin'].tolist()
            bins.sort(key=histogram_residuals.get, reverse=True)
            split = [bins]
            for i in range(self.args.max_leaves-1):
                max_gain = 0
                max_split = split.copy()
                for idx, parent in enumerate(split):
                    sim_parent = self.get_sim_score(parent, histogram_residuals, histogram_hessian)
                    # if len(parent) == 1 no more split
                    for j in range(len(parent)-1):
                        left_split = parent[0:j+1]
                        right_split = parent[j+1:]
                        # sim = [[num_res, sum_res]]
                        # get_sim_score_numerical(left_split, sim) => get sim_score from sim list
                        sim_left = self.get_sim_score(left_split, histogram_residuals, histogram_hessian)
                        sim_right = self.get_sim_score(right_split, histogram_residuals, histogram_hessian)
                        gain = sim_left + sim_right - sim_parent
                        if max_gain < gain:
                            max_gain = gain
                            copied_split = split.copy()
                            del copied_split[idx]
                            copied_split.insert(idx, left_split)
                            copied_split.insert(idx+1, right_split)
                            max_split = copied_split
                if max_gain == 0:
                    break
                split = max_split.copy()
        return split

    def fit(self):
        self.preprocess()
        # print('Training...')
        self.candidate_feature = self.hist_columns + []
        self.min_cf = int(len(self.candidate_feature) * self.args.min_cf)
        self.output_values = np.zeros_like(self.residuals, dtype=float)
        self.consumed_eps = 0
        self.consumed_mu_2 = 0
        # Laplace
        if self.args.privacy:
            if self.args.delta == 0:
                self.eps_per_epoch = self.ebm_eps / self.args.epochs
            # Gaussian
            else:
                self.mu_2_per_epoch = self.ebm_mu_2 / self.args.epochs

        for epoch in range(self.args.epochs):
            # stop if there is no candidate_feature
            if len(self.candidate_feature) == 0:
                break
            self.additive_terms.append({})
            mean_scores = {}
            remove_features = []
            num_data_splits = {}

            # re-calculate noise_scale from remain privacy budget
            if self.args.privacy:
                if self.args.delta == 0:
                    self.residual_noise_scale = len(self.candidate_feature) / self.eps_per_epoch
                else:
                    self.residual_noise_scale = np.sqrt(len(self.candidate_feature) / self.mu_2_per_epoch)

            for feature in self.candidate_feature:
                self.additive_terms[epoch][feature] = {}
                self.additive_terms[epoch][feature]['additive_term'] = []

                num_data_splits[feature] = []
                mean_score = 0.
             
                # get best split
                if self.data_type[feature] == NUMERICAL: # numerical
                    histogram_residuals = self.get_histogram_residual(feature)
                    histogram_hessian = self.get_histogram_hessian(feature)
                    best_splits = self.get_split_numerical(feature, histogram_residuals, histogram_hessian)
                    # best_splits = [[0], [1, 2], [3, 4, 5], [6]]
                    self.additive_terms[epoch][feature]['split'] = best_splits
                    # print(feature)
                    # print(self.histograms[feature]['count'])
                    # print(histogram_residuals)
                    # print(f'best_splits: {best_splits}')

                    for split in best_splits:
                        avg_residuals = 0.
                        sum_residuals = 0.
                        sum_hessian = 0.
                        for bin in split:
                            sum_residuals += histogram_residuals[bin]
                            sum_hessian += histogram_hessian[bin]
                        num_data_splits[feature].append(sum_hessian)
                        # noise to residual
                        if self.args.privacy:
                            if self.args.delta == 0:
                                sum_residuals += np.random.laplace(0., self.residual_noise_scale * self.range_label)
                            else:
                                sum_residuals += np.random.normal(0., self.residual_noise_scale * self.range_label)

                        # Laplace, (e, 0)-dp
                        if self.args.delta == 0:
                            mean_score += abs(sum_residuals)
                        # Gaussian, (e, d)-dp
                        else:
                            mean_score += (sum_residuals ** 2) / sum_hessian

                        # assert that num_hist > 0
                        if sum_hessian >= 1:
                            avg_residuals = sum_residuals / sum_hessian
                            update_grad = avg_residuals * self.lr
                        else:
                            update_grad = 0.
                        self.additive_terms[epoch][feature]['additive_term'].append(update_grad)

                        for bin in split:
                            # update ouput function (decision function) f
                            self.decision_function[feature][bin] += update_grad
                            # update residuals
                            if self.args.regression:
                                for idx in self.hist_idx[feature][bin]:
                                    self.residuals[idx] -= update_grad
                            else:
                                for idx in self.hist_idx[feature][bin]:
                                    self.output_values[idx] += update_grad
                                    # self.residuals[idx] = self.label[idx] -1 + (1/(1+np.exp(self.output_values[idx])))

                else: # categorical
                    histogram_residuals = self.get_histogram_residual(feature)
                    histogram_hessian = self.get_histogram_hessian(feature)
                    best_splits = self.get_split_categorical(feature, histogram_residuals, histogram_hessian)
                    # best_splits = [['0'], ['1', '3]', ['2', '4', '5]', ['6']]
                    self.additive_terms[epoch][feature]['split'] = best_splits
                    
                    for split in best_splits:
                        avg_residuals = 0.
                        sum_residuals = 0.
                        sum_hessian = 0.
                        for bin in split:
                            sum_residuals += histogram_residuals[bin]
                            sum_hessian += histogram_hessian[bin]
                        num_data_splits[feature].append(sum_hessian)
                        if self.args.privacy:
                            if self.args.delta == 0:
                                # b or lambda = self.residual_noise_scale * self.range_label
                                sum_residuals += np.random.laplace(0., self.residual_noise_scale * self.range_label)
                            else:
                                sum_residuals += np.random.normal(0., self.residual_noise_scale * self.range_label)

                        # Laplace, (e, 0)-dp
                        if self.args.delta == 0:
                            mean_score += abs(sum_residuals)
                        # Gaussian, (e, d)-dp
                        else:
                            mean_score += (sum_residuals ** 2) / sum_hessian

                        if sum_hessian >= 1:
                            avg_residuals = sum_residuals / sum_hessian
                            update_grad = avg_residuals * self.lr
                        else:
                            update_grad = 0.
                        

                        self.additive_terms[epoch][feature]['additive_term'].append(update_grad)

                        for bin in split:
                            # update ouput function (decision function) f
                            self.decision_function[feature][bin] += update_grad
                            # update residuals
                            if self.args.regression:
                                for idx in self.hist_idx[feature][bin]:
                                    self.residuals[idx] -= update_grad
                            else:
                                for idx in self.hist_idx[feature][bin]:
                                    self.output_values[idx] += update_grad
                                    # self.residuals[idx] = self.label[idx] -1 + (1/(1+np.exp(self.output_values[idx])))

                mean_scores[feature] = mean_score
                if not (self.args.regression):
                    self.residuals = self.label -1 + (1/(1+np.exp(self.output_values)))
            if self.args.privacy:
                if self.args.delta == 0:
                    self.consumed_eps += self.eps_per_epoch
                else:
                    self.consumed_mu_2 += self.mu_2_per_epoch

            # adaptive feature
            # if self.args.adaptive_feature:
            if self.min_cf > 0:
                if self.args.adaptive_feature and self.min_cf < len(self.candidate_feature):
                    mean_scores = {k: v for k, v in sorted(mean_scores.items(), key=lambda item: item[1])}
                    for k, v in mean_scores.items():
                        af_threshold = self.get_af_threshold(k, epoch, num_data_splits[k])
                        
                        if v < af_threshold:
                            remove_features.append(k)

                    for r in remove_features:
                        if self.min_cf < len(self.candidate_feature):
                            self.candidate_feature.remove(r)
                            print(f'removed feature: {r} at epoch {epoch}')
            else:
                if self.args.adaptive_feature:
                    mean_scores = {k: v for k, v in sorted(mean_scores.items(), key=lambda item: item[1])}
                    for k, v in mean_scores.items():
                        af_threshold = self.get_af_threshold(k, epoch, num_data_splits[k])
                        
                        if v < af_threshold:
                            remove_features.append(k)

                    for r in remove_features:
                        self.candidate_feature.remove(r)
                        # print(f'removed feature: {r} at epoch {epoch}')
                    
                    if len(self.candidate_feature):
                        self.candidate_feature = self.hist_columns + []


        # intercept
        for feature in self.hist_columns:
            if self.data_type[feature] == NUMERICAL:
                mean_score = 0.
                for bin in range(len(self.histograms[feature]['count'])):
                    mean_score += self.decision_function[feature][bin] * self.histograms[feature]['count'][bin]
                mean_score = mean_score / self.total_data
                for bin in range(len(self.histograms[feature]['count'])):
                    self.decision_function[feature][bin] -= mean_score
                self.intercept += mean_score
            else:
                mean_score = 0.
                for bin in self.histograms[feature]['bin']:
                    mean_score += self.decision_function[feature][bin] * self.histograms[feature]['count'][bin]
                mean_score = mean_score / self.total_data
                for bin in self.histograms[feature]['bin']:
                    self.decision_function[feature][bin] -= mean_score
                self.intercept += mean_score
        
        return

    def explain(self):
        # global explain for feature importance
        global_score = []
        global_features = self.hist_columns + []
        for feature in self.hist_columns:
            abs_mean_score = 0
            if self.data_type[feature] == NUMERICAL:
                for bin in range(len(self.histograms[feature]['count'])):
                    abs_mean_score += abs(self.decision_function[feature][bin]) * self.histograms[feature]['count'][bin]
            else:
                for bin in self.histograms[feature]['bin']:
                    abs_mean_score += abs(self.decision_function[feature][bin]) * self.histograms[feature]['count'][bin]
            abs_mean_score = abs_mean_score / self.total_data
            global_score.append(abs_mean_score)
        sorted_g_score = np.sort(global_score)
        sort_idx = np.argsort(global_score)
        sorted_features = [global_features[i] for i in sort_idx]
        plt.figure(figsize=(15,10))
        plt.barh(sorted_features, sorted_g_score)
        plt.xlabel('importance score')
        plt.ylabel('Feature')
        plt.show()
        
        # global explain shape function

        global_score = []
        global_features = self.hist_columns + []
        for feature in self.hist_columns:
            if self.data_type[feature] == NUMERICAL:
                x = []
                y = []
                for bin in range(len(self.histograms[feature]['count'])):
                    y.append(self.decision_function[feature][bin])
                    y.append(self.decision_function[feature][bin])
                    if bin == len(self.histograms[feature]['count'])-1:
                        x.append(self.histograms[feature]['bin'][bin])
                        x.append(2*self.histograms[feature]['bin'][bin] - self.histograms[feature]['bin'][bin-1])
                    else:
                        x.append(self.histograms[feature]['bin'][bin])
                        x.append(self.histograms[feature]['bin'][bin+1])
                plt.figure(figsize=(15,10))
                plt.axhline(y=0.0, color='r', linestyle='-')
                plt.plot(x,y)
                plt.xlabel(feature)
                plt.ylabel('shape function')

            else:
                feature_values = []
                shape = []
                for bin in self.histograms[feature]['bin']:
                    feature_values.append(bin)
                    shape.append(self.decision_function[feature][bin])
                sorted_shape = np.sort(shape)
                sort_idx = np.argsort(shape)
                sorted_fv = [feature_values[i] for i in sort_idx]
                plt.figure(figsize=(15,10))
                plt.axvline(x=0.0, color='r', linestyle='-')
                plt.barh(sorted_fv, sorted_shape)
                plt.xlabel(feature)
                plt.ylabel('Feature values')
            
            plt.show()

        # local explain for train data

        # for feature in self.hist_columns:


        

    def predict(self, df, label_df):
        # print('Predicting...')
        num_data = df.shape[0]
        output_value = np.zeros(num_data) + self.intercept
        label = label_df.to_numpy().astype(float)
        # replace values to dpother
        if self.args.privacy:
            for i in self.hist_columns:
                if self.data_type[i] == CATEGORICAL:
                    col_data = df[i].to_numpy()
                    dpothers = []
                    uniq_vals, _ =np.unique(col_data, return_inverse=True)
                    for u in uniq_vals:
                        if u not in self.histograms[i]['bin']:
                            dpothers.append(u)
                    df[i] = df[i].replace(dpothers, 'DPOther')

        # regression or
        # classification
        for i in self.hist_columns:
            col_data = df[i].to_numpy()
            # numeric
            if self.data_type[i] == NUMERICAL:
                for idx, val in enumerate(col_data):
                    prev_left = 0
                    for j in range(len(self.histograms[i]['bin'])):
                        if self.histograms[i]['bin'][j] <= val:
                            prev_left = j
                        else:
                            break
                    output_value[idx] += self.decision_function[i][prev_left] 

            # categorical
            elif self.data_type[i] == CATEGORICAL:
                for idx, val in enumerate(col_data):
                    if val in self.histograms[i]['bin']:
                        output_value[idx] += self.decision_function[i][val] 
        
        if self.args.regression:
            y_hat = output_value
            total_squared_error = np.sum((label - y_hat)**2)
            mse = total_squared_error / num_data

            return math.sqrt(mse)

        else:
            total_loss = 0
            total_correct = 0
            y_hat = 1 / (1 + np.exp(-output_value))
            for idx, l in enumerate(label):
                if l == 0:
                    # total_loss -= np.log(1-y_hat[idx])
                    if y_hat[idx] < 0.5:
                        total_correct += 1
                else:
                    # total_loss -= np.log(y_hat[idx])
                    if y_hat[idx] >= 0.5:
                        total_correct += 1

            mean_loss = total_loss / num_data

            auroc = roc_auc_score(label, y_hat)
            # cls_pred = (y_hat[:] >= 0.5).astype(bool)
            # print(y_hat)
            # print(cls_pred)
            # f1 = f1_score(label, cls_pred)

            return total_correct / num_data, auroc
            # return f1, auroc