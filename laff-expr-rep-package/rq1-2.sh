#!/bin/sh

source codepath.sh
cd "$Path"

python3 set_config.py -run_alg true -run_eval true

echo "MFM Sequential Filling"
python3 set_config.py -alg mfm -type all -order seq
python3 alg_exection.py

echo "MFM Random Filling"
python3 set_config.py -alg mfm -type all -order rand
python3 alg_exection.py

echo "ARM Sequential Filling"
python3 set_config.py -alg arm -type all -order seq
python3 alg_exection.py

echo "ARM Random Filling"
python3 set_config.py -alg arm -type all -order rand
python3 alg_exection.py

echo "Naive Sequential Filling"
python3 set_config.py -alg naive -type all -order seq
python3 alg_exection.py

echo "Naive Random Filling"
python3 set_config.py -alg naive -type all -order rand
python3 alg_exection.py

echo "FLS Sequential Filling"
python3 set_config.py -alg fls -type all -order seq
python3 alg_exection.py

echo "FLS Random Filling"
python3 set_config.py -alg fls -type all -order rand
python3 alg_exection.py

echo "LAFF Sequential Filling"
python3 set_config.py -alg laff -type all -order seq -use_local true -use_filter true
python3 alg_exection.py

echo "LAFF Random Filling"
python3 set_config.py -alg laff -type all -order rand -use_local true -use_filter true
python3 alg_exection.py
 
python3 set_config.py -eval_type all -eval_order seq rand -eval_algs mfm arm naive fls laff 
python3 rq_effectiveness.py

python3 statistical_tests.py