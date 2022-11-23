import matplotlib.pyplot as plt

# fig = plt.figure()

# ax = fig.add_subplot(1,1,1)
fig, ax = plt.subplots()

X = ['', '0.01', '0.02', '0.05', '0.1', '0.2', '0.5']
ax.set_xticklabels(X)

y0 = [0.732122992,0.732122992
,0.732122992
,0.732122992
,0.732122992
,0.732122992
]

y1 = [67.55273145,
56.31637857,
34.94708647,
27.39428298,
18.62530267,
11.46618964,
]

y2 = [9.352305635,
7.794771291,
5.026486399,
3.879803937,
3.027850133,
2.016362443
]

plt.plot(y0, marker='x', label='EBM')
plt.plot(y1, marker='H', label='DP-EBM')
plt.plot(y2, marker='^', label='DP-EBM-LFP')

ax.set_xlabel('epsilon')
ax.set_ylabel('RMSE')
ax.set_title('Wine Regression')
ax.legend()
plt.show()