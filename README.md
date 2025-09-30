# Galaxy_Morphology_Final


The dataset consists of 3 components.
1. Images folder - contains 243,437 images of galaxies (Source: https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images)
2. gz2_filename_mapping.csv - contains mappings from image file name to a unique ID (Source: https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images)
3. gz2_hart16.csv - the primary dataset containing data on votes cast by citizen scientists (Source: https://data.galaxyzoo.org/) [Hart et al.(2016): https://mnras.oxfordjournals.org/content/461/4/3663]

<img width="764" height="781" alt="image" src="https://github.com/user-attachments/assets/298f69e1-5966-4b17-b580-8d1cf8e9df1f" />


/home/shubham.agarwal_phd24/Galaxy_Classification
|-- 1_Data
|   |-- processed
|   |   `-- final_galaxy_test_set.csv
|   `-- raw
|       |-- gz2_filename_mapping.csv
|       |-- gz2_hart16.csv
|       `-- images
|-- 2_Scripts
|   |-- Model_1_final.py
|   |-- Model_2_final.py
|   |-- Model_2_sub.py
|   |-- __pycache__
|   |   |-- Model_1_final.cpython-38.pyc
|   |   |-- Model_2_final.cpython-38.pyc
|   |   `-- Model_2_sub.cpython-38.pyc
|   |-- evaluate.py
|   |-- evaluate2.py
|   `-- inspect_batch.py
|-- 3_Jobs
|   |-- eval_model_1.sh
|   |-- eval_model_2.sh
|   |-- run_debug_model_1.sh
|   |-- run_model_1.sh
|   `-- run_model_2.sh
|-- 4_Models
|   |-- Model_1
|   |   |-- checkpoints
|   |   `-- final_stable_cnn_model.pth
|   `-- Model_2
|       |-- checkpoints
|       `-- final_stable_cnn_model.pth
|-- 5_Results
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
`-- Conda_Envs
    `-- galaxy_env
        |-- bin
        |-- compiler_compat
        |-- conda-meta
        |-- include
        |-- lib
        |-- man
        |-- share
        |-- ssl
        |-- x86_64-conda-linux-gnu
        `-- x86_64-conda_cos7-linux-gnu

27 directories, 31 files
