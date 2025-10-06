# Galaxy_Morphology_Final


The dataset consists of 3 components.
1. Images folder - contains 243,437 images of galaxies (Source: https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images)
2. gz2_filename_mapping.csv - contains mappings from image file name to a unique ID (Source: https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images)
3. gz2_hart16.csv - the primary dataset containing data on votes cast by citizen scientists (Source: https://data.galaxyzoo.org/) [Hart et al.(2016): https://mnras.oxfordjournals.org/content/461/4/3663]

For ease of understanding, the structure of directories and files which were used for training and testing the models, as it was on the HPC, is given below:

```plaintext
.
`-- Galaxy_Classification
    |-- 1_Data  # Contains all raw and processed datasets.
    |   |-- processed
    |   |   `-- final_galaxy_test_set.csv
    |   `-- raw
    |       |-- gz2_filename_mapping.csv
    |       |-- gz2_hart16.csv
    |       `-- images  # Raw galaxy images (ignored by tree for brevity).
    |-- 2_Scripts  # All Python scripts for model training and evaluation.
    |   |-- Model_1_final.py
    |   |-- Model_2_final.py
    |   |-- Model_2_sub.py
    |   |-- evaluate.py
    |   |-- evaluate2.py
    |   `-- inspect_batch.py
    |-- 3_Jobs  # Shell scripts for submitting and running jobs on a server/HPC.
    |   |-- eval_model_1.sh
    |   |-- eval_model_2.sh
    |   |-- run_debug_model_1.sh
    |   |-- run_model_1.sh
    |   `-- run_model_2.sh
    |-- 4_Models  # Saved model weights and checkpoints.
    |   |-- Model_1
    |   |   `-- checkpoints
    |   `-- Model_2
    |       `-- checkpoints
    |-- 5_Results  # Output files, logs, and performance plots from model runs.
    |   |-- Model_1
    |   |   |-- err_evaluate.log
    |   |   |-- err_model_1.log
    |   |   |-- final_performance_plot.png
    |   |   |-- out_evaluate.log
    |   |   |-- out_model_1.log
    |   |   `-- train_loss_history.npy
    |   `-- Model_2
    |       |-- err_evaluate2.log
    |       |-- err_model_2.log
    |       |-- final_performance_plot.png
    |       |-- out_evaluate2.log
    |       |-- out_model_2.log
    |       `-- train_loss_history.npy
    |-- Conda_Envs   # Environment configuration files (e.g., .yml files).
    `-- check_files.py  # Utility script to verify the project setup.

16 directories, 27 files
```

<img width="3000" height="1800" alt="training_loss_comparison" src="https://github.com/user-attachments/assets/97816d20-3dc3-414a-af18-39893c42fc27" />

