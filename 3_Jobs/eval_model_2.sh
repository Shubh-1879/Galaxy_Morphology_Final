#!/bin/bash

#PBS -N model_2__eval
#PBS -l ncpus=32
#PBS -o /home/shubham.agarwal_phd24/Galaxy_Classification/5_Results/Model_2/out_evaluate2.log
#PBS -e /home/shubham.agarwal_phd24/Galaxy_Classification/5_Results/Model_2/err_evaluate2.log
#PBS -q gpu
#PBS -M shubham.agarwal_phd24@ashoka.edu.in

module load compiler/anaconda3
python3 /home/shubham.agarwal_phd24/Galaxy_Classification/2_Scripts/evaluate2.py

echo "Job has finished."
