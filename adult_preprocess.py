import pandas as pd

df = pd.read_csv('data/adult_label_str.csv')
df['label'] = df['label'].replace(['>50K', '<=50K'], [1, 0])
df.to_csv('data/adult.csv', index=False)