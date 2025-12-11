#!/bin/bash

for instance in 0 50 100 150 200 250; do
    step=50
    sum=$((instance + step))
    python rl_cvrp.py --config original_rl_100.yml --train 0 --evaluate 1 --instances-eval ${instance}:${sum}&
done
