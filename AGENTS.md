# Contributor Guide

## Project Information
This project implements a mining system for the Bittensor subnet Gradients.
A miner in this subnet receives requests (tasks) to finetune large language models.
These tasks can be one of SFT, DPO, or GRPO finetuning.

A RunPod serverless worker runs the docker image defined in serverless.dockerfile.

This docker image is populated by the files it needs to finetune the models, these files are training/hpo_optuna.py and training/train.py
The miner runs hpo_optuna.py which does a hyperparameter search to find optimal parameters for the full training run and then launches the full training script with the found parameters.

The task includes the number of hours we have to complete the finetuning.

All the miners compete to get the lowest eval_loss in the given time, or highest eval_loss if it is a GRPO task.

