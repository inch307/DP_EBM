$lr_o = 0.1
For ($i=1;$i-lt11;$i++) {
	$lr = $lr_o * $i
	python .\run.py --label label --lr $lr --epochs 300 --max_leaves 3  --privacy --split_strategy --eps 8.0 --delta 0
}
