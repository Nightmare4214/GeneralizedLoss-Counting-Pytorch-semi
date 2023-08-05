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
python preprocess_shanghai.py --origin_dir PATH_TO_ORIGIN_DATASET --data_dir PATH_TO_DATASET --part 'A/B'
```

[//]: # (The dataset can be constructed followed by [Bayesian Loss]&#40;https://github.com/ZhihengCV/Bayesian-Crowd-Counting&#41;.)

## Test

```bash
python test.py --data_dir PATH_TO_DATASET --save_dir PATH_TO_CHECKPOINT --dataset "qnrf/sha/shb"
```

## Train

```bash
python train.py --data_dir PATH_TO_DATASET --save_dir PATH_TO_CHECKPOINT --dataset "qnrf/sha/shb" --max_epoch xxx --cost "p_norm" --p_norm 2 --phi "KL" --extra_aug --scheduler "poly/linear"
```

## Reproduction

### UCF-QNRF

paper: L2 + KL + epsilon=0.01: mae: 83.3, mse: 142.3


| p_norm | norm  | blur | mae  | mse  |
|--------|-------|------| ---- | ---- |
| 1      | True  | 0.01 | 95.79747431863568    | 172.58023273939637     |
| 2      | True  | 0.01 | 82.89548871617117    | 144.92945336885356     |

### Shanghai-A

paper: L2 + KL + epsilon=0.01: mae: 58.1, mse: 95.9

| p_norm | norm  | blur | mae  | mse  |
|--------|-------|------| ---- | ---- |
| 2      | True  | 0.01 | 60.95482723529522    | 94.12666011754312   |

### Shanghai-B

paper: L2 + KL + epsilon=0.01: mae: 6.5, mse: 10.2

| p_norm | norm  | blur | mae  | mse  |
|--------|-------|------| ---- | ---- |
| 2      | True  | 0.01 | 7.564692005326476    | 12.813679879205807   |

### Acknowledgement
We use [GeomLoss](https://www.kernel-operations.io/geomloss/) package to compute transport matrix. Thanks for the authors for providing this fantastic tool. The code is slightly modified to adapt to our framework.
