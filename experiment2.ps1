$eps = 0.1, 0.5, 1.0, 2.0, 4.0, 8.0
$delta = 0, 1e-6
$split_strategy = "", "--split_strategy"
$max_leaves = 3, 4, 8, 16
$adaptive_feature = "", "--adaptive_feature"
$af_prob = 0.01, 0.02, 0.05, 0.1, 0.2
$af_count = 1, 2, 4
$adaptive_lr = "", "--adaptive_lr"

foreach ($e in $eps) {
    foreach ($d in $delta){
        foreach ($ml in $max_leaves){
            foreach ($ss in $split_strategy){
                foreach ($ap in $af_prob){
                    foreach ($ac in $af_count){
                            python .\run.py --label label --lr 0.01 --epochs 300 --max_leaves $ml --privacy $ss --eps $e --delta $d --n_runs 25 --adaptive_feature --af_prob $ap --af_count $ac
                    }
                }
            }
        }
    }
}


foreach ($e in $eps) {
    foreach ($d in $delta){
        foreach ($ml in $max_leaves){
            foreach ($ss in $split_strategy){
                foreach ($ap in $af_prob){
                    foreach ($ac in $af_count){
                            python .\run.py --label label --lr 0.01 --epochs 300 --max_leaves $ml --privacy $ss --eps $e --delta $d --n_runs 25 --adaptive_feature --adaptive_lr --af_prob $ap --af_count $ac
                    }
                }
            }
        }
    }
}