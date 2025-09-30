#!/bin/bash

#PBS -N model_debug
#PBS -o out_debug.log
#PBS -e err_debug.log
#PBS -q cpu
#PBS -l ncpus=1
#PBS -M shubham.agarwal_phd24@ashoka.edu.in
#PBS -m abe

# --- DEBUGGING SCRIPT ---
echo "--- STARTING DEBUG SCRIPT ---"
echo "Job is running on host: $(hostname)"
echo "Job running as user: $(whoami)"

# Define the full path to the Python executable we want to test
MY_PYTHON="/home/shubham.agarwal_phd24/.conda/envs/galaxy_env/bin/python"

echo "--- CHECK 1: Verifying Python Executable ---"
echo "Checking for Python at path: ${MY_PYTHON}"
if [ -f "$MY_PYTHON" ]; then
    echo "SUCCESS: Python executable found."
    ls -l "$MY_PYTHON"
else
    echo "CRITICAL FAILURE: Python executable NOT FOUND at this path."
    echo "--- DEBUG SCRIPT FINISHED ---"
    exit 1
fi

echo "--- CHECK 2: Checking Python Version ---"
"$MY_PYTHON" --version

echo "--- CHECK 3: Attempting to import cv2 ---"
"$MY_PYTHON" -c "import cv2; print('>>> SUCCESS: Python can import cv2!')"

echo "--- CHECK 4: Listing installed packages from within the job ---"
"$MY_PYTHON" -m pip list

echo "--- DEBUG SCRIPT FINISHED ---"

