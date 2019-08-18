# GCN-GAN

This repository provide a pytorch implemention for the GCN-GAN model proposed in "A Non-linear Temporal Link Prediction Model for Weighted Dynamic Networks" INFOCOM 2019, [[pdf]][1].

## Quick Start

Modify hyper-parameters in file ```config.yml```.

### Preprocess data

```
python preprocess.py
```

### Train

```
python train.py
```

### Test

```
python test.py
```

[1]: https://arxiv.org/pdf/1901.09165.pdf