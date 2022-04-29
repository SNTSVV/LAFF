#!/bin/sh

source codepath.sh
cd "$Path"

python3 set_config.py -run_alg true -run_eval true

echo "LAFF Sequential Filling"
python3 set_config.py -alg laff -type all -order seq -use_local true -use_filter true
python3 alg_exection.py

echo "LAFF-nofilter Sequential Filling"
python3 set_config.py -alg laff-nofilter -type all -order seq -use_local true -use_filter false
python3 alg_exection.py

echo "LAFF-nolocal Sequential Filling"
python3 set_config.py -alg laff-nolocal -type all -order seq -use_local false -use_filter true
python3 alg_exection.py

echo "LAFF-noboth Sequential Filling"
python3 set_config.py -alg laff-noboth -type all -order seq -use_local false -use_filter false
python3 alg_exection.py

echo "LAFF Random Filling"
python3 set_config.py -alg laff -type all -order rand -use_local true -use_filter true
python3 alg_exection.py

echo "LAFF-nofilter Random Filling"
python3 set_config.py -alg laff-nofilter -type all -order rand -use_local true -use_filter false
python3 alg_exection.py

echo "LAFF-nolocal Random Filling"
python3 set_config.py -alg laff-nolocal -type all -order rand -use_local false -use_filter true
python3 alg_exection.py

echo "LAFF-noboth Random Filling"
python3 set_config.py -alg laff-noboth -type all -order rand -use_local false -use_filter false
python3 alg_exection.py

python3 set_config.py -eval_type all -eval_order seq rand -eval_algs laff-noboth laff-nofilter laff-nolocal laff
python3 rq_variant.py
