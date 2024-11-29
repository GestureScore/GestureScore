# GestureScore

GestureScore

This directory provides the scripts for quantitative evaluation of our gesture generation framework. We currently support the following measures:

- Average Jerk and Acceleration (AJ)
- Histogram of Moving Distance (HMD) for velocity and acceleration
- Hellinger distance between histograms
- Canonical Correlation Analysis (CCA) coefficient
- Fréchet Gesture Distance (FGD)

## FID Score

### Processing Data:

```bash
python data_processing.py --real_path="./data/zeggs/real" --predict_path="./data/zeggs/predict" --gpu=mps
```

### Training Embedding:

```bash
python training_embedding.py --dataset=../data/real_dataset.npz --gpu=mps --epoch=1
```

### Evaluation FGD Score

```bash
cd FGD/
python evaluate_FGD.py -real=../data/real_dataset.npz -predict ../data/predict_dataset.npz --gpu=mps
```



## Average Jerk and Acceleration (AJ)

```bash
calk_jerk_or_acceleration.py
```

## Histogram of Moving Distance (HMD) for velocity and acceleration

```bash
calc_histogram.py
```

## Hellinger distance between histograms

```bash
hellinger_distance.py
```

## Canonical Correlation Analysis (CCA) coefficient

```bash
calc_cca.py
```

## Fréchet Gesture Distance (FGD)

```bash
calc_fgd.py
```
