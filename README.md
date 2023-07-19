# Learning to count via unbalanced optimal transport

unofficial implement

## Data

Dowload Dataset UCF-QNRF [Link](https://www.crcv.ucf.edu/data/ucf-qnrf/)

## Data preparation

```
python preprocess_dataset.py --origin_dir PATH_TO_ORIGIN_DATASET --data_dir PATH_TO_DATASET
```
[//]: # (The dataset can be constructed followed by [Bayesian Loss]&#40;https://github.com/ZhihengCV/Bayesian-Crowd-Counting&#41;.)

## Test

```
python test.py --data_dir PATH_TO_DATASET --save_dir PATH_TO_CHECKPOINT
```

## Train

```
python train.py --data_dir PATH_TO_DATASET --save_dir PATH_TO_CHECKPOINT
```

## Reproduction
paper: L2 + KL + epsilon=0.01: mae: 83.3, mse: 142.3


| p_norm | norm  | blur | mae  | mse  |
|--------|-------|------| ---- | ---- |
| 1      | True  | 0.01 | 95.79747431863568    | 172.58023273939637     |
| 2      | True  | 0.01 | 85.76067790870896    | 155.0306857078346     |


### Acknowledgement
We use [GeomLoss](https://www.kernel-operations.io/geomloss/) package to compute transport matrix. Thanks for the authors for providing this fantastic tool. The code is slightly modified to adapt to our framework.
