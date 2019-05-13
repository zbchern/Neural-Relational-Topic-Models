# Neural Relational Topic Models for Scientific Article Analysis

Code for CIKM 2018 paper [Neural Relational Topic Models for Scientific Article Analysis](https://dl.acm.org/citation.cfm?id=3271696)

## Prerequisities
* Python version: 2.7
* Tensorflow version: 1.4.1

## How to run
* The script for running this model with dataset citeulike-a is *main_citeulike_a.py*, while *main_cora.py* is for running with dataset cora.
* To pretrain the model, set the mode as 1 in main file.
* After pretraining, set the mode as 2 to finetune the model.

## Remarks
* The checkpoint files generated from pretraining process have been provided.
* The log files recording the experiment results of the model are also given.
