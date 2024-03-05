#!/bin/bash --login

/home/dsg2060/.lmod.d
# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate <YOUR_ENV_NAME>

## REQUIRES CHILD SCRIPT OR WILL LOOP INDEFINITELY ##

# Create a text file
echo "This is an dummy file. We will delete it to signal when inference finishes." > red_light.txt

# submit inference job
sbatch <YOUR_gpt4_eval_child.sh_SCRIPT> 

# Check if the dummy script exists; loop until it doesn't.
while true; do
  if [ -f red_light.txt ]; then
    sleep 5
  else
    echo "Green light!\n"
    break
  fi
done

# execute python script; do not change python file path/content
python3 /grphome/grp_retnet/compute/evaluation_metrics/gpt4_eval.py ../../configs/user_configs/<YOUR_CONFIG_HERE>.yaml
