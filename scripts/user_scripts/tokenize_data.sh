# Execute script with the names of the splits to tokenize, separated by spaces
# Split names: train, validation, test

# Define a function to run the Python command with a given argument
run_command() {
    local arg="$1"
    python3 \
        ../../src/tokenize_data.py \
        ../../configs/user_configs/example_compute_config.yaml \
        "$arg" &
    pid=$!
    echo "Started process with PID $pid for argument: $arg"
    pids+=("$pid")
}

# Initialize an array to store process IDs
pids=()

# Check if arguments are provided
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided."
    exit 1
fi

# Run the Python command for each argument in parallel
for arg in "$@"; do
    run_command "$arg"
done

# Wait for all processes to finish
for pid in "${pids[@]}"; do
    wait "$pid"
    echo "Process with PID $pid has finished."
done

echo "All processes have finished."
