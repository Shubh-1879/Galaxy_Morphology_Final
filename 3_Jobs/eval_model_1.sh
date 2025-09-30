#!/bin/bash

#PBS -N model_1__eval
#PBS -l ncpus=32
#PBS -o /home/shubham.agarwal_phd24/Galaxy_Classification/5_Results/Model_1/out_evaluate.log
#PBS -e /home/shubham.agarwal_phd24/Galaxy_Classification/5_Results/Model_1/err_evaluate.log
#PBS -q gpu
#PBS -M shubham.agarwal_phd24@ashoka.edu.in

module load compiler/anaconda3
python3 /home/shubham.agarwal_phd24/Galaxy_Classification/2_Scripts/evaluate.py

echo "Job has finished."
