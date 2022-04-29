#!/bin/sh

source codepath.sh
cd "$Path"

python3 set_config.py -run_alg true -run_eval true

echo "MFM partial filling"
python3 set_config.py -alg mfm -type partial -order rand
python3 alg_exection.py

echo "ARM partial filling"
python3 set_config.py -alg arm -type partial -order rand
python3 alg_exection.py

echo "LAFF partial filling"
python3 set_config.py -alg laff -type partial -order rand -use_local true -use_filter true
python3 alg_exection.py

python3 set_config.py -eval_type partial -eval_order rand -eval_algs mfm arm laff
python3 rq_number_filled_fields.py