$eps = 0.1, 0.5, 1.0, 2.0, 4.0, 8.0
$delta = 0, 1e-6
$privacy = "", "--privacy"
$split_strategy = "", "--split_strategy"
$max_leaves = 3, 4, 8, 16, 32
$adaptive_feature = "", "--adaptive_feature"
$af_prob = 0.001, 0.005, 0.01, 0.02
$af_max_remove = 1, 2, 5
$max_bins = 32, 64
For ($i=1;$i-lt11;$i++) {
	$lr = $lr_o * $i
	python .\run.py --label label --lr 0.01 --epochs 300 --max_leaves 3  --privacy --split_strategy --eps 8.0 --delta 0
}
