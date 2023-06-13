# DistRepLearning

Main code repository for experiments conducted for distributed representation learning, a fourth year final year project in electrical and electronic engineering at Imperial College London.

## Use
Each script in the main directory is run as a standalone experiment. Arguments to be passed can be seen in each script's execution block. An example for running the fully collaborative decentralised autoencoder experiment on CIFAR-10 with localised testing would be:

```
python dist_autoencoder.py --model_training collaborative \
--model_epochs 60 \
--encoded_dim 32 \
--classifier_training collaborative \
--classifier epochs \
--testing local \
--dataset CIFAR \
--output ./results \
--optimizer AdamW \
--topology random \
--scheduler
```
