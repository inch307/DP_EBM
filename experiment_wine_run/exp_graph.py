import matplotlib.pyplot as plt
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--path')
parser.add_argument('--af_prob', type=float, help='start_af_prob')
parser.add_argument('--eps', type=float)
parser.add_argument('--delta', type=float)
parser.add_argument('--num_eps', type=int, default=6)

args = parser.parse_args()

if args.path in ['adult', 'telco', 'syn_cls']:
    Y = 'acc'
    ylabel = 'accuracy'
    title = args.path + ' classification'

elif args.path in ['wine', 'cal_housing', 'syn_reg']:
    Y = 'rmse'
    ylabel = 'rmse'
    title = args.path + ' regression'

if args.delta == 0:
    dpebmfp = 'DP-EBM-LFP'
else:
    dpebmfp = 'DP-EBM-GFP'

original_eps_lst = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
start_eps = original_eps_lst.index(args.eps)
eps_lst = []

for i in range(args.num_eps):
    eps_lst.append(original_eps_lst[start_eps + i])

# df = pd.read_csv('experiment_' + args.path +'_final.csv')
df = pd.read_csv(args.path + '.csv')
df=df.dropna(how='all')
print(df)

# EBM
y_ebm = df.loc[df['privacy'] == False, Y].values[0]

y0 = [y_ebm for i in range(args.num_eps)]

# DPEBM
y1 = []
for e in eps_lst:
    y1.append(df.loc[ (df['cv'] == 0) & (df['eps'] == e) & (df['delta'] == args.delta) & (df['privacy'] == True) & (df['af'] == False), Y].values[0])

# DPEBMFP
y2 = []
for e in eps_lst:
    y2.append(df.loc[ (df['cv'] == 0) & (df['eps'] == e) & (df['delta'] == args.delta) & (df['privacy'] == True) & (df['af'] == True) & (df['af_prob']==args.af_prob), Y].values[0])

fig, ax = plt.subplots()

X = ['']
for e in eps_lst:
    X.append(str(e))
# X = ['', '0.01', '0.02', '0.05', '0.1', '0.2', '0.5']
ax.set_xticklabels(X)

# y0 = [0.844698296
# , 0.844698296
# ,0.844698296
# ,0.844698296
# ,0.844698296
# ,0.844698296
# ]

# y1 = [0.524900967,
# 0.514097958,
# 0.611183786,
# 0.649488715,
# 0.704621526,
# 0.767240903
# ]

# y2 = [0.584904038,
# 0.616784892,
# 0.6899002,
# 0.748650392,
# 0.759975434,
# 0.759054199
# ]

plt.plot(y0, marker='x', label='EBM')
plt.plot(y1, marker='H', label='DP-EBM')
plt.plot(y2, marker='^', label=dpebmfp)

ax.set_xlabel('epsilon')
ax.set_ylabel(ylabel)
ax.set_title(title)
ax.legend()
plt.show()