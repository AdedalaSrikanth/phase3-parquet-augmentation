phase 3 parquet augmentation project

this project builds parquet to parquet data augmentation software for tabular datasets. the software tracks the original source row for every augmented sample and supports both normal and group based randomization.

main features

- reads input parquet file from same directory
- saves output parquet file in same directory
- tracks source_row_id for each augmented row
- supports normal randomization
- supports group based randomization
- marks is_augmented, augmentation_round and randomization_mode
- supports minority only augmentation to balance dataset

files

- build_input.py builds parquet input from csv
- build_gallstone_input.py builds parquet input using gallstone dataset
- phase2_parquet_augment.py performs main augmentation
- phase3_analysis.py analyzes class distribution and augmentation lineage
- phase3_model_comparison.py compares original, normal and group based datasets
- phase3_minority_augment.py augments only minority class
- phase3_minority_model_test.py evaluates minority balanced dataset

input files

- input.parquet
- gallstone.csv
- ai4i2020.csv

output files

- augmented_minority_only.parquet
- model_comparison_results.csv
- minority_model_results.csv

how to run

activate virtual environment

run input build
python build_gallstone_input.py

run augmentation
python phase2_parquet_augment.py

run analysis
python phase3_analysis.py

run model comparison
python phase3_model_comparison.py

run minority augmentation
python phase3_minority_augment.py

run model test
python phase3_minority_model_test.py

final results

original, normal augmented and group based datasets showed high accuracy around 0.96 but f1 score was 0. this means model was predicting only majority class.

after applying minority only augmentation, dataset became balanced and model performance changed to

accuracy around 0.59  
f1 score around 0.59  

this shows that accuracy alone is misleading for imbalanced datasets and minority balancing improves meaningful performance.

note on smote

smote type methods were considered but they require two samples for generating new data which makes source tracking difficult. this project uses single sample augmentation for simplicity.

project status

phase 3 completed