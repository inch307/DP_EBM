$eps = 0.01, 0.1, 0.5, 1, 2, 4, 8
$delta = 0, 1e-6
$split_strategy = "", "--split_strategy"
$max_leaves = 3
$adaptive_feature = "", "--adaptive_feature"
$af_prob = 0.01, 0.02
$af_count = 2
$data_path = "data/adult.csv"
$label = "label"

#foreach ($ml in $max_leaves){
#    python .\run.py --data_path $data_path --label $label --lr 0.01 --epochs 300 --max_leaves $ml --n_runs 25
#}

foreach ($e in $eps) {
    foreach ($d in $delta){
        foreach ($ml in $max_leaves){
            foreach ($ss in $split_strategy){
                python .\run.py --data_path $data_path --label $label --lr 0.01 --epochs 300 --max_leaves $ml --privacy $ss --eps $e --delta $d --n_runs 25
            }
        }
    }
}

foreach ($e in $eps) {
        foreach ($ml in $max_leaves){
                foreach ($ap in $af_prob){
                    foreach ($ac in $af_count){
                        python .\run.py --data_path $data_path --label $label --lr 0.01 --epochs 300 --max_leaves $ml --privacy --eps $e --delta 0 --n_runs 25 --adaptive_feature --adaptive_lr --af_prob $ap --af_count $ac
                    }
                }
        }
}

foreach ($e in $eps) {
        foreach ($ml in $max_leaves){
            foreach ($ss in $split_strategy){
                foreach ($ap in $af_prob){
                    foreach ($ac in $af_count){
                        python .\run.py --data_path $data_path --label $label --lr 0.01 --epochs 300 --max_leaves $ml --privacy $ss --eps $e --delta 1e-6 --n_runs 25 --adaptive_feature --adaptive_lr --af_prob $ap --af_count $ac
                    }
                }
            }
        }
}