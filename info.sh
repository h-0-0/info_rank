#!/bin/bash

# Function to get CPU information
get_cpu_info() {
    echo "=== CPU Information ==="
    lscpu | grep -E 'Model name|Socket|Core|Thread|CPU\(s\)|CPU MHz'
    echo "Total CPU Memory: $(free -h | grep Mem | awk '{print $2}')"
    echo
}

# Function to get GPU information
get_gpu_info() {
    echo "=== GPU Information ==="
    
    # Check if nvidia-smi is available for NVIDIA GPUs
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader
    else
        echo "No NVIDIA GPUs found or nvidia-smi is not installed."
    fi
    
    echo
}

# Execute the functions
get_cpu_info
get_gpu_info