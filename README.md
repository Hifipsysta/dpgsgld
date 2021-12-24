## DP-GSGLD

Code for DP-GSGLD

### About Questions

Any suggestions/questions is welcome.

### Requirements

- torch  1.9.0

- matplotlib 3.4.1

- scipy 1.6.2

- numpy 1.19.2

### Experiments for DLG

    cd DP-GSGLD-DLG
    python CIFAR.py --learning_rate 0.01 --device cuda0 --iterations 300 --img_index 7 --scale 1 --epochs 2 --grad_clip 1 --optimizer FDPGSGLD

### Experiments for IG

    cd DP-GSGLD-IG
    python reconstruct_image.py --model ResNet20-4 --dataset LFW --trained_model --cost_fn sim --indices def --restarts 1 --save_image --target_id -1 --max_iterations 24000 --scale 5 --optimizer DPSGD


