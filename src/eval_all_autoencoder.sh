#!/bin/bash
echo "Bash version ${BASH_VERSION} - Evaluate all models..."

for config in 'autoencoder'; do

  echo "Store baseline $config..."
  python main_evaluation.py $config --load ../checkpoints/$config.ckpt --noise 0 --line_interrupt 0 --store_baseline_activations_path ../tmp/$config.pt
  sleep 2

  echo "Evaluate different noise for $config..."
  for noise in $(seq 0.0 .01 0.2); do
    python main_evaluation.py $config --load ../checkpoints/$config.ckpt --noise $noise --line_interrupt 0 --load_baseline_activations_path ../tmp/$config.pt --wandb
  done

  echo "Evaluate different line interrupts for $config..."
  for li in {1..7}; do
    python main_evaluation.py $config --load ../checkpoints/$config.ckpt --noise 0 --line_interrupt $li --load_baseline_activations_path ../tmp/$config.pt --wandb
  done

done
