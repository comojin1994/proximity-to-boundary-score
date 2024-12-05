# A Novel Adversarial Approach for EEG Dataset Refinement: Enhancing Generalization through Proximity-to-Boundary Scoring

This repository is the official implementations of Proximity-to-Boundary Score in pytorch-lightning style:

```text
TBA
```

<!-- ```text
S.-J. Kim, H. Kong, D.-H. Lee, H.-G. Kwak, and S.-W. Lee, "A Novel Adversarial Approach for EEG Dataset Refinement: Enhancing Generalization through Proximity-to-Boundary Scoring," IEEE Transactions on Cybernetics, 2025. (Accepted)
``` -->

## Overall Framework

![Alt text](docs/architecture.png)

## Abstract

> As deep learning performs remarkably in pattern recognition from complex data, it is used to interpret user intentions from electroencephalography (EEG) signals. However, deep learning models trained from EEG datasets have low generalization ability owing to numerous noisy samples in datasets. Therefore, pioneer research has focused on distinguishing and eliminating noisy samples from datasets. One intuitive solution is based on the property of noisy samples during the training phase. Noisy samples are located near the decision boundary after model training. Therefore, they can be detected using a gradient-based adversarial attack. However, limitations of usability exist because the intuitive solution requires additional hyperparameter optimizations, resulting in a trade-off between accurateness and efficiency. In this paper, we proposed a novel training framework that enhances the generalization ability of the model by reducing the influence of noisy samples during training, without additional hyperparameter optimizations. We designed the proximity-to-boundary score (PBS) to continuously measure the data closeness to the decision boundary. The proposed framework improved the generalization ability of the model across two motor imagery datasets and one sleep stage dataset. We qualitatively confirmed that data with low PBS are indeed noisy samples and degrade the model training. Hence, we demonstrated that employing the proposed framework accurately and efficiently mitigates the influence of noisy samples, enhancing the model's generalization capabilities.

## Algorithm of the proposed framework

<p align="center">
  <img src="docs/algorithm.png" alt="Algorithm Image" width="50%">
</p>

## 1. Installation

### 1.1 Clone this repository

```bash
$ git clone https://github.com/comojin1994/proximity-to-boundary-score.git
```

### 1.2 Environment setup

> Create docker container and `databases` and `logs` directory by under script

```bash
$ cd docker
$ make start.train
$ docker exec -it torch-train bash
$ cd proximity-to-boundary-score
```

### 1.3 Preparing data

> After downloading the [BCI Competition IV 2a & 2b](https://www.bbci.de/competition/iv/#download) and [Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/), revise the data's directory in the `datasets/setups/{dataset}.py` files

```bash
# We provide the official code using BCI Competition IV 2a.
$ make setup
```

```python
BASE_PATH = {Dataset directory}
SAVE_PATH = {Revised dataset directory}
```

## 2. Quantitative Analysis

### 2.1 Comparison of performances in motor imagery classification datasets

| Method                 |   |          |         BCIC2a       |                |   |          |      BCIC2b          |                |
|------------------------|:-:|:--------------:|:--------------:|:--------------:|:-:|:--------------:|:--------------:|:--------------:|
|                        |   |     EEGNet     |   DeepConvNet  | ShallowConvNet |   |     EEGNet     |   DeepConvNet  | ShallowConvNet |
| Baseline               |   | 61.51±0.96 | 56.97±0.48 | 59.55±0.75 |   | 77.24±0.36 | 76.44±0.22 | 76.77±0.26 |
| Random Dropout         |   | 61.79±0.34 | 56.74±0.38 | 59.26±0.54 |   | 77.05±0.26 | 76.42±0.16 | 76.45±0.20 |
| MC Dropout             |   | 62.16±0.56 | 62.24±0.34 | 63.44±0.37 |   | 77.27±0.20 | 79.55±0.07 | 78.58±0.27 |
| Influence Score        |   | 62.48±1.18 | 61.64±1.01 | 64.09±0.67 |   | 78.66±0.17 | **80.34±0.03** | 80.20±0.19 |
| Forgetting Score       |   | 61.85±0.78 | 59.62±0.01 | 60.09±0.55 |   | 77.89±0.25 | 80.28±0.01 | 77.40±0.29 |
| DRAA                   |   |            |            |            |   |            |            |            |
| $\alpha = \text{1e-3}$ |   | 63.54±0.95 | 56.71±0.42 | 60.90±0.71 |   | 78.27±0.35 | 79.45±1.06 | 79.25±0.18 |
| $\alpha = \text{1e-5}$ |   | 63.91±0.35 | 62.16±0.54 | 62.70±0.40 |   | 78.35±0.14 | 79.80±0.22 | 80.03±0.21 |
| $\alpha = \text{1e-7}$ |   | 64.68±0.43 | 62.59±0.14 | 63.35±1.16 |   | 78.62±0.33 | 80.27±0.18 | **80.98±0.29** |
| **Proposed**           |   | **64.90±0.46** | **62.67±0.47** | **64.31±1.00** |   | **78.67±0.23** | 80.31±0.14 | 80.41±0.16 |

### 2.2 Comparison of performances in sleep stage classification datasets

<p align="center">
  <img src="docs/sleepedf.png" width="50%">
</p>

### 2.3 Comparison of efficiency

| Method                 |                Complexity               | $\tau$ opt. | GPU Min. |
|------------------------|:---------------------------------------:|:-----------:|:--------:|
| Influence Score        |    $\mathcal{O}(ME+Mp^2+p^3+M^2p^2)$    |    ✓   |  116.63  |
| MC Dropout             |     $\mathcal{O}(ME+MS_\text{MC}f)$     |    ✓   |   0.42   |
| Forgetting Score       |            $\mathcal{O}(ME)$            |    ✓   |     -    |
| DRAA                   |                                         |        |          |
| $\alpha = \text{1e-3}$ | $\mathcal{O}(ME+M\bar{S}_\text{1e-3}f)$ |    ✓   |   0.05   |
| $\alpha = \text{1e-5}$ | $\mathcal{O}(ME+M\bar{S}_\text{1e-5}f)$ |    ✓   |    6.2   |
| $\alpha = \text{1e-7}$ | $\mathcal{O}(ME+M\bar{S}_\text{1e-7}f)$ |    ✓   |  839.15  |
| **Proposed**           |          **$\mathcal{O}(ME+MSf)$**      |    𐄂   |   **1.48**   |

## 3. Qualitative Analysis

### 3.1 Signal visualization

<p align="center">
  <img src="docs/signal_visualization.png">
</p>

### 3.2 Feature visualization

<p align="center">
  <img src="docs/feature_visualization.png">
</p>

## 4. User Manual

### STAGE 1: Training model $f$ using $\Chi_{train}$

```yaml
# config.yaml

mode: "all"

litmodel: "base"

...

score: null

...

weighted_loss: False

```

```bash
$ make train
```

### STAGE 2: Evaluate the proximity of data to the boundary

> Run `evaluation.ipynb` jupyter notebook

> Revise the options in config files

```python
# Block [2] in evaluation.ipynb

args = get_configs()
args = init_configs(args)
init_settings(args)

args.WEIGHT_PATH = "{checkpoint path}"
```

### STAGE 3: Get $\hat{\theta}$ utilizing PBS-guided soft rejection

```yaml
# config.yaml

mode: "cls"

litmodel: "weighted"

...

score: "pbs"

...

weighted_loss: True

```

```bash
$ make train
```