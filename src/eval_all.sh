#!/bin/bash
echo "Bash version ${BASH_VERSION} - Evaluate all models..."

for config in 'net-fragments'; do

  for act_threshold in 0.3 0.5 0.7; do

    for square_factor in '0.6 0.8 1.0 1.2 1.4 1.6' '1.2 1.4 1.6 1.8 2.0 2.2' '1.8 1.9 2.0 2.1 2.2 2.3'; do

      echo "Store baseline $config..."
      python main_evaluation.py $config --load ../checkpoints/$config.ckpt --noise 0 --line_interrupt 0 --store_baseline_activations_path ../tmp/$config.pt --act_threshold $act_threshold --square_factor $square_factor
      sleep 2

      echo "Evaluate different noise for $config..."
      for noise in $(seq 0.0 .01 0.2); do
        python main_evaluation.py $config --load ../checkpoints/$config.ckpt --noise $noise --line_interrupt 0 --load_baseline_activations_path ../tmp/$config.pt --act_threshold $act_threshold --square_factor $square_factor --wandb
      done

      echo "Evaluate different line interrupts for $config..."
      for li in {1..7}; do
        python main_evaluation.py $config --load ../checkpoints/$config.ckpt --noise 0 --line_interrupt $li --load_baseline_activations_path ../tmp/$config.pt --act_threshold $act_threshold --square_factor $square_factor --wandb
      done

    done
  done
done
