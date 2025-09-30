#!/bin/bash

#PBS -N model_1_final
#PBS -l ncpus=32
#PBS -o /home/shubham.agarwal_phd24/Galaxy_Classification/5_Results/Model_1/out_model_1.log
#PBS -e /home/shubham.agarwal_phd24/Galaxy_Classification/5_Results/Model_1/err_model_1.log

#PBS -q gpu
#PBS -M shubham.agarwal_phd24@ashoka.edu.in

module load compiler/anaconda3
python3 /home/shubham.agarwal_phd24/Galaxy_Classification/2_Scripts/Model_1_final.py

echo "Job has finished."