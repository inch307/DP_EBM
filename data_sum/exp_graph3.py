import matplotlib.pyplot as plt
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--task')
# parser.add_argument('--af_prob', type=float, help='start_af_prob')
# parser.add_argument('--eps', type=float)
parser.add_argument('--delta', type=float)

args = parser.parse_args()

# if args.path in ['adult', 'telco', 'syn_cls']:
if args.task == 'cls':
    Y = 'acc'
    ylabel = 'Accuracy'
    path1 = 'experiment_adult_final.csv'
    path2 = 'experiment_telco_final.csv'
    path3 = 'experiment_syn_cls_final.csv'
    title1 = '(a) Adult'
    title2 = '(b) Telco'
    title3 = '(c) Syn_cls'
    legend = 'lower right'

# elif args.path in ['wine', 'cal_housing', 'syn_reg']:
elif args.task == 'reg':
    Y = 'rmse'
    ylabel = 'RMSE'
    path1 = 'experiment_wine_final.csv'
    path2 = 'experiment_cal_housing_final.csv'
    path3 = 'experiment_syn_reg_final.csv'
    title1 = '(a) Wine'
    title2 = '(b) Cal_housing'
    title3 = '(c) Syn_reg'
    legend = 'upper right'

if args.delta == 0:
    dpebmfp = 'DP-EBM-LFP'
else:
    dpebmfp = 'DP-EBM-GFP'

plt.rcParams['font.family'] = 'Arial'

original_eps_lst = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13,4))

########################################### Modify here ###############################################
start_eps1 = original_eps_lst.index(0.01)
start_eps2 = original_eps_lst.index(0.01)
start_eps3 = original_eps_lst.index(0.2)

num_eps1 = 6
num_eps2 = 6
num_eps3 = 6

af_prob1 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
af_prob2 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
af_prob3 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
########################################### Modify here ###############################################

########################################### exp1 ###############################################

eps1_lst = []
for i in range(num_eps1):
    eps1_lst.append(original_eps_lst[start_eps1 + i])

df = pd.read_csv(path1)
df = df.dropna(how='all')

# EBM
y_ebm = df.loc[df['privacy'] == False, Y].values[0]

y0 = [y_ebm for i in range(num_eps1)]

# DPEBM
y1 = []
for e in eps1_lst:
    y1.append(df.loc[ (df['eps'] == e) & (df['delta'] == args.delta) & (df['privacy'] == True) & (df['af'] == False), Y].values[0])

# DPEBMFP
y2 = []
for e in eps1_lst:
    i = 0
    y2.append(df.loc[ (df['eps'] == e) & (df['delta'] == args.delta) & (df['privacy'] == True) & (df['af'] == True) & (df['af_prob']==af_prob1[i]), Y].values[0])
    i += 1

ax1.plot(y0, marker='x', label='EBM')
ax1.plot(y1, marker='H', label='DP-EBM')
ax1.plot(y2, marker='^', label=dpebmfp)

########################################### exp1 ###############################################

########################################### exp2 ###############################################

eps2_lst = []
for i in range(num_eps2):
    eps2_lst.append(original_eps_lst[start_eps2 + i])

df = pd.read_csv(path2)
df = df.dropna(how='all')

# EBM
y_ebm = df.loc[df['privacy'] == False, Y].values[0]

y0 = [y_ebm for i in range(num_eps2)]

# DPEBM
y1 = []
for e in eps2_lst:
    y1.append(df.loc[ (df['eps'] == e) & (df['delta'] == args.delta) & (df['privacy'] == True) & (df['af'] == False), Y].values[0])

# DPEBMFP
y2 = []
for e in eps2_lst:
    i = 0
    y2.append(df.loc[ (df['eps'] == e) & (df['delta'] == args.delta) & (df['privacy'] == True) & (df['af'] == True) & (df['af_prob']==af_prob2[i]), Y].values[0])
    i += 1

ax2.plot(y0, marker='x', label='EBM')
ax2.plot(y1, marker='H', label='DP-EBM')
ax2.plot(y2, marker='^', label=dpebmfp)

########################################### exp2 ###############################################

########################################### exp3 ###############################################

eps3_lst = []
for i in range(num_eps3):
    eps3_lst.append(original_eps_lst[start_eps3 + i])

df = pd.read_csv(path3)
df = df.dropna(how='all')

# EBM
y_ebm = df.loc[df['privacy'] == False, Y].values[0]

y0 = [y_ebm for i in range(num_eps1)]

# DPEBM
y1 = []
for e in eps3_lst:
    y1.append(df.loc[ (df['eps'] == e) & (df['delta'] == args.delta) & (df['privacy'] == True) & (df['af'] == False), Y].values[0])

# DPEBMFP
y2 = []
for e in eps3_lst:
    i = 0
    y2.append(df.loc[ (df['eps'] == e) & (df['delta'] == args.delta) & (df['privacy'] == True) & (df['af'] == True) & (df['af_prob']==af_prob3[i]), Y].values[0])
    i += 1

ax3.plot(y0, marker='x', label='EBM')
ax3.plot(y1, marker='H', label='DP-EBM')
ax3.plot(y2, marker='^', label=dpebmfp)

########################################### exp3 ###############################################

# plot

X1 = ['']
for e in eps1_lst:
    X1.append(str(e))

X2 = ['']
for e in eps2_lst:
    X2.append(str(e))

X3 = ['']
for e in eps3_lst:
    X3.append(str(e))

ax1.set_xticklabels(X1)
ax2.set_xticklabels(X2)
ax3.set_xticklabels(X3)

ax1.set_xlabel('epsilon')
ax1.set_ylabel(ylabel)
# ax1.set_title(title1)
# ax1.legend(loc=legend)


# box = ax2.get_position()
# ax2.legend(loc='upper center', bbox_to_anchor=(1.04, 1))
ax2.legend(bbox_to_anchor=(-0.25, 1.04, 1.5, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)

ax2.set_xlabel('epsilon')
ax2.set_ylabel(ylabel)
# ax2.set_title(title2)
# ax2.legend(loc=legend)

ax3.set_xlabel('epsilon')
ax3.set_ylabel(ylabel)
# ax3.set_title(title3)
# ax3.legend(loc=legend)
plt.subplots_adjust(wspace=0.3)
plt.show()