# Galaxy_Morphology_Final

The purpose of this project was to classify images of extra-terrestrial bodies into categories posed by the Galaxy Zoo 2 project using CNNs. I collected a few images myself using a telescope (cool work, haha - see below), but the bulk of images were sourced as listed below.

The dataset consists of 3 components.
1. Images folder - contains 243,437 images of galaxies (Source: https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images)
2. gz2_filename_mapping.csv - contains mappings from image file name to a unique ID (Source: https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images)
3. gz2_hart16.csv - the primary dataset containing data on votes cast by citizen scientists (Source: https://data.galaxyzoo.org/) [Hart et al.(2016): https://mnras.oxfordjournals.org/content/461/4/3663]

## Final report

The final report for this project can be found here: https://drive.google.com/file/d/12QvdyD1S0XSNWscHeF3APO3RPx9Q873R/view?usp=sharing
You can write to me at shubham.agarwal_phd24@ashoka.edu.in or shubhamagarwal1879@gmail.com if you are not able to access it.

## Directory structure
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

## Fun images

Besides this project, here are some of the fun images we took using Celestron NexStar 8SE (a telescope at Ashoka Physics lab) during coursework:

<img width="650" height="691" alt="image" src="https://github.com/user-attachments/assets/7a4b4dfc-920c-42c6-8e95-a3f10ec36cb7" />

<img width="786" height="564" alt="image" src="https://github.com/user-attachments/assets/2aaa0de0-e861-4772-8c15-6543d1e170b1" />

<img width="791" height="579" alt="image" src="https://github.com/user-attachments/assets/fe778833-0b5d-499c-9d63-35d0b2358abb" />

<img width="826" height="581" alt="image" src="https://github.com/user-attachments/assets/6a1c07af-9669-4376-92b9-749deca54065" />

<img width="774" height="574" alt="image" src="https://github.com/user-attachments/assets/56c8c857-33ae-48b8-9820-c81243529eb4" />

<img width="794" height="567" alt="image" src="https://github.com/user-attachments/assets/09799c00-0b5a-40ac-9e84-c813baa1ae1f" />





