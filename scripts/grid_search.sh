#!/bin/bash

# Loop with custom step size; not parallelized!
for i in $(seq 0 <grid_search_size> 1); do  # (seq <START> <STOP> <STEP_SIZE>)
  python3 ../../src/grid_search.py ../../configs/user_configs/<YOUR_CONFIG_HERE>.yaml i
done
