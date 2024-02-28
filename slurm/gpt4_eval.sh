#!/bin/bash --login

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate <YOUR_ENV_NAME>

# Create a text file
echo "This is an dummy file. We will delete it to signal when inference finishes." > red_light.txt

# submit inference job
# TODO: *NOTE: Be sure to append `rm red_light.txt` to the end of your generate bash script!!
sbatch <YOUR_GENERATE_JOB_HERE> ../../configs/user_configs/<YOUR_CONFIG_HERE>.yaml

# Check if the dummy script exists; loop until it doesn't.
while true; do
  if [ -f red_light.txt ]; then
    sleep 5
  else
    echo "Green light!\n"
    break
  fi
done

# execute python script
python3 /grphome/grp_retnet/compute/evaluation_metrics/gpt4_eval.py ../../configs/user_configs/<YOUR_CONFIG_HERE>.yaml
