# Net Fragments

This repository contains the code of "Biologically Inspired Net Fragments in Machine Learning Systems: From Theory to Implementation".
Please note that the original code used to produce the results in the paper is available at [https://github.com/sagerpascal/lateral-connections](https://github.com/sagerpascal/lateral-connections).
However, this repository contains a more recent version of the code, which is more modular and easier to use.
It achieves about the same results as the original code, but it is not guaranteed that the results are exactly the same.


## Setup
Create conda environment

```bash
conda create --name net-fragments python=3.10
```

Activate Environment

```bash
conda activate net-fragments
```

Install requirements

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Run experiments


```bash
python main_training.py <config> --wandb --plot --store <store_path>
python main_evaluation.py <config> --load <store_path>
```

Training configurations:
- `train_v1`: only 2 timesteps, slightly higher square factors, without Bernoulli neurons
- `train_v2`: 6 timesteps, without Bernoulli neurons
- `train_v3`: 6 timesteps, with Bernoulli neurons
