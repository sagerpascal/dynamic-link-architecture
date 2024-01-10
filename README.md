# Net Fragments

This repository contains the code of "Biologically Inspired Net Fragments in Machine Learning Systems: From Theory to Implementation".
Please note that the original code is available at [https://github.com/sagerpascal/lateral-connections](https://github.com/sagerpascal/lateral-connections), containing much more options (including a feedback layer S3 and more datasets).
However, this repository contains a more recent version of the code, which is more modular and easier to use.
All results in the paper have been re-created using this code.


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

Training can be done as follows:

```bash
python main_training.py <config> --wandb --plot --store <store_path>
```

where `config` is the name of the training configuration (see below), `store_path` is the path where the results should be stored, and `wandb` and `plot` are optional flags to enable logging to [wandb](https://wandb.ai/) and plotting of the results, respectively.

A concrete example would be:

```bash
python main_training.py train_v1 --plot --wandb --store train_v1.ckpt 
````


Training configurations:
- `train_v1`: S1 activates based on a fixed threshold, S2 uses only 2 timesteps, has slightly higher square factors, and does not use Bernoulli neurons
- `train_v2`: S1 activates based on a fixed threshold, S2 uses 6 timesteps and does not use Bernoulli neurons
- `train_v3`: S1 activates based on a fixed threshold, S2 uses 6 timesteps and Bernoulli neurons
- `train_v4`: S1 uses Bernoulli neurons, S2 uses 6 timesteps and Bernoulli neurons
- `train_v5`: S1 does not have an activation function (jus clip activations below 0 to 0), S2 uses 6 timesteps and Bernoulli neurons


For evaluation, first create store the baseline activations of the trained model:


```bash
python main_evaluation.py $config --load <ckpt_path> --noise 0 --line_interrupt 0 --store_baseline_activations_path <baseline_path>

```

Then run the evaluation:

```bash
python main_evaluation.py $config --load <ckpt_path> --noise <noise_float> --line_interrupt <interrupt_int> --load_baseline_activations_path <baseline_path> --act_threshold <act_threshold> --square_factor <square_factor> --wandb
```

where `noise` is the noise level, `line_interrupt` is the number of interrupted lines, `act_threshold` is the activation threshold, and `square_factor` is the square factor.
You can also use `--act_threshold bernoulli` to wir with Bernoulli neurons. However, in this case, the results are based
on randomness and will vary between runs. Therefore, we recommend using a fixed activation threshold, e.g. `--act_threshold 0.5`, which will also makes the plot easier to comprehend.