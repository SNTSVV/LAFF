

## Prerequisites

- Python3
- [jq](https://stedolan.github.io/jq/)



## Steps

1. To be able to run our tool you need to install the different python packages present in the file "requirements.txt ". 
```bash
cd lacquer-expr-rep-package
pip install -r requirements.txt
```
2. To be able to evaluate our tool, you need to unzip the file bio_human_version_1.0.xml.zip in the raw folder, and run the script xml2csv.py to convert the xml file to a csv dataset
```bash
cd dataset/raw
python xml2csv.py
```
3. To get the results of all research questions, you need to run the makefile 
```bash
cd lacquer-expr-rep-package
make 
```
4. If you want to run only a specific script, you can run it by typing Make and the specified script name (rq1-2, rq3, rq4, and rq5)
```bash
cd lacquer-expr-rep-package
make  rq1-2
```

We also provide a Docker image. After installing the required libraries for docker, the artifact can be obtained by running the command ``docker pull laffs/laff:latest''.

## Expected Output

- When we run the code, first it preprocesses the raw data, and then generates some pre-processed datasets and a folder ``train-test``, which includes: 
	- The training set (preprocessed training set which encodes the value of fields into IDs): ``ncbi-homo-sapien_train.csv``
	- The testing sets for sequential and random filling scenarios: ``ncbi-homo-sapiens-test_rand.csv`` and  ``ncbi-homo-sapiens-test_seq.csv``
	- The mapping value/id represent the mapping between the value of the fields and the corresponding id: ``ncbi-homo-sapiens_val_id.csv`` 
    - When we run LAFF, it trains models and saved the models in the folder ``train-test/model``
- After running the code, it generates the detailed results for each setting (eg. MFM random, LAFF random, etc) which is saved in the folder ``train-test/results/details``.
This file saves the details of the prediction on each testing instance. The csv contains the following columns: target (which represents the current target of the instance), truth (the true value of the target), ranked (the ranked list), remained (means if the suggestion will be shown to the user), train and predict (training and prediction time).
 
For example, after executing 
```bash
make  rq1-2
```
it is expected to see files named as ``ncbi-homo-sapiens_seq_[alg]_eval.csv`` and ``ncbi-homo-sapiens_rand_[alg]_eval.csv``, where [alg] is the name of algorithms including laff, arm, mfm, fls, and naive.
