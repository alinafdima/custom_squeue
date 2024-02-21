# Custom squeue
Custom squeue for a slurm server

Parses the output of `scontrol show job -d` and displays it, similar to `squeue`.

It features an automatic GPU count.


## Example usage:
```` bash
python custom_squeue.py --nodes alpha:8 beta:16 gamma:4
````


## Example entry in .bashrc
```` bash
alias gpus="watch -n 2 python \
    <PATH_TO_FOLDER>/custom_squeue.py \
    --nodes \
        alpha:8 \
        beta:16 \
        gamma:4 \
    "
````