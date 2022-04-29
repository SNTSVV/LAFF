#!/bin/sh

source codepath.sh
cd "$Path"

python3 set_config.py -run_alg true -run_eval true

echo "LAFF sample sequential filling"
python3 set_config.py -alg laff -type sample -round 10 -order seq -use_local true -use_filter true
python3 alg_exection.py

echo "LAFF sample random filling"
python3 set_config.py -alg laff -type sample -round 10 -order rand -use_local true -use_filter true
python3 alg_exection.py

python3 set_config.py -eval_type sample -eval_order seq rand -eval_algs laff 
python3 rq_size_of_data.py