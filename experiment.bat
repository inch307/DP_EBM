python ./run.py --data_path data/telco.csv --label Churn --lr 0.01 --epochs 300 --max_leaves 3 --eps 0 --delta 0 --af_prob 0 --n_runs 10
python ./run.py --data_path data/telco.csv --label Churn --lr 0.01 --epochs 300 --max_leaves 3 --eps 1 --delta 0 --n_runs 10 --adaptive_feature --af_prob 0.1 --fake_eps 4 --privacy
pause