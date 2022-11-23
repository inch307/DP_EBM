import matplotlib.pyplot as plt

# fig = plt.figure()

# ax = fig.add_subplot(1,1,1)
fig, ax = plt.subplots()

X = ['', '0.01', '0.02', '0.05', '0.1', '0.2', '0.5']
ax.set_xticklabels(X)

y0 = [0.844698296
, 0.844698296
,0.844698296
,0.844698296
,0.844698296
,0.844698296
]

y1 = [0.524900967,
0.514097958,
0.611183786,
0.649488715,
0.704621526,
0.767240903
]

y2 = [0.584904038,
0.616784892,
0.6899002,
0.748650392,
0.759975434,
0.759054199
]

plt.plot(y0, marker='x', label='EBM')
plt.plot(y1, marker='H', label='DP-EBM')
plt.plot(y2, marker='^', label='DP-EBM-LFP')

ax.set_xlabel('epsilon')
ax.set_ylabel('accuracy')
ax.set_title('Adult classification')
ax.legend()
plt.show()