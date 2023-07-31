# Learning to count via unbalanced optimal transport

unofficial implement

## Data

Dowload Dataset UCF-QNRF [Link](https://www.crcv.ucf.edu/data/ucf-qnrf/)  
Download Shanghai Tech Part A and Part B[Link](https://www.kaggle.com/tthien/shanghaitech)  
Download NWPU[Link](https://www.crowdbenchmark.com/nwpucrowd.html)  

## Data preparation

### UCF-QNRF
```bash
python preprocess_dataset.py --origin_dir PATH_TO_ORIGIN_DATASET --data_dir PATH_TO_DATASET
```

### Shanghai Tech

```bash
python preprocess_shanghai.py --origin_dir PATH_TO_ORIGIN_DATASET --data_dir PATH_TO_DATASET --part 'A'
```

[//]: # (The dataset can be constructed followed by [Bayesian Loss]&#40;https://github.com/ZhihengCV/Bayesian-Crowd-Counting&#41;.)

## Test

```bash
python test.py --data_dir PATH_TO_DATASET --save_dir PATH_TO_CHECKPOINT
```

## Train

```bash
python train.py --data_dir PATH_TO_DATASET --save_dir PATH_TO_CHECKPOINT
```

## Reproduction

### UCF-QNRF

paper: L2 + KL + epsilon=0.01: mae: 83.3, mse: 142.3


| p_norm | norm  | blur | mae  | mse  |
|--------|-------|------| ---- | ---- |
| 1      | True  | 0.01 | 95.79747431863568    | 172.58023273939637     |
| 2      | True  | 0.01 | 85.76067790870896    | 155.0306857078346     |

### Shanghai-A

paper: L2 + KL + epsilon=0.01: mae: 58.1, mse: 95.9

| p_norm | norm  | blur | mae  | mse  |
|--------|-------|------| ---- | ---- |
| 2      | True  | 0.01 | 68.1625064598335    | 99.64657828156318   |

### Shanghai-B

paper: L2 + KL + epsilon=0.01: mae: 6.5, mse: 10.2

| p_norm | norm  | blur | mae  | mse  |
|--------|-------|------| ---- | ---- |
| 2      | True  | 0.01 | 7.78297430955911    | 12.895112162260704   |

### Acknowledgement
We use [GeomLoss](https://www.kernel-operations.io/geomloss/) package to compute transport matrix. Thanks for the authors for providing this fantastic tool. The code is slightly modified to adapt to our framework.
