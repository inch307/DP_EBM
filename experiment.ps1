$eps = 0.01, 0.02, 0.05, 0.07, 0.1
$zeps = 0.1, 1, 2, 4, 8
$delta = 1e-6
$cls_hes = "", "--classification_hessian"
$max_leaves = 3
$adaptive_feature = "", "--adaptive_feature"
$af_prob = 0.01, 0.02, 0.05, 0.1
$af_count = 1
$data_path = "data/adult.csv"
$label = "label"

python .\run.py --data_path $data_path --label $label --lr 0.01 --epochs 300 --max_leaves $ml --n_runs 25 --classification_hessian
python .\run.py --data_path $data_path --label $label --lr 0.01 --epochs 300 --max_leaves $ml --n_runs 25

# foreach ($e in $eps) {
#     foreach ($d in $delta){
#         foreach ($ml in $max_leaves){
#             foreach ($ss in $split_strategy){
#                 python .\run.py --data_path $data_path --label $label --lr 0.01 --epochs 300 --max_leaves $ml --privacy $ss --eps $e --delta $d --n_runs 25
#             }
#         }
#     }
# }

foreach ($e in $zeps) {
        foreach ($cls in $cls_hes){
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