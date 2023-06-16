﻿# **O**pen**L2D**

## Abstract

Public resource limitations have significantly hindered the development and benchmarking of learning to defer (L2D) algorithms, which aim to optimally combine human and AI capabilities in hybrid decision-making systems. In such systems, human availability and domain-specific concerns introduce complexity, while obtaining human predictions for training and evaluation is costly. To overcome these challenges, we introduce OpenL2D, a novel framework designed to generate synthetic expert decisions and testbed settings for L2D methods. OpenL2D facilitates the creation of synthetic experts with adjustable bias and feature dependence, simulates realistic human work capacity constraints, and provides diverse training and testing scenarios. To demonstrate its utility, we employ OpenL2D on a public fraud detection dataset, synthesizing a team of 50 fraud analysts, and we benchmark L2D baselines under an array of 220 distinct testing scenarios. We believe that OpenL2D will serve as a pivotal instrument in facilitating a systematic, rigorous, reproducible, and transparent evaluation and comparison of L2D methods, thereby fostering the development of more synergistic human-AI collaboration in decision-making systems.

## Resources
In this repo, we provide users with:

* Code for use of our framework.
* [Datasets and models](https://drive.google.com/drive/folders/1nAUlxdOmwC6ZNtch3rxwKwNUVrYNmxkV) used in our benchmark.

The submitted version of the paper, the appendix, and the Datasheet are available in the following links:

* [Paper](documents/paper.pdf)
* [Appendix](documents/appendix.pdf)
* [Datasheet](documents/datasheet.pdf)

## Using the OpenL2D Fraud Detection Dataset

In our experiments, for training and testing of assignment methods, we utilized the **OpenL2D Fraud Detection Dataset**. This dataset is comprised of:

* An Input Dataset.
* Synthetic Expert prediction table.
* Dataset with limited expert predictions.
* Sets of capacity constraint tables.

For more information on each of these components, please consult the provided [Datasheet](documents/datasheet.pdf).

* ### Step 1: Download the Code in this repo:
The sets of capacity constraint tables and the synthetic expert prediction table are generated by using the input dataset and a set of configs included in this repo. For easy use of our dataset and available notebooks, we encourage users to download the repo in its entirety.

* ### Step 2: Download the Input Dataset
Our input dataset is the base variant of the Bank Account Fraud Tabular Dataset, available [here](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022?resource=download&select=Base.csv). This dataset should then be placed in the folder "/data".

* ### Step 3: Download the Models, Dataset with limited expert predictions and other necessary data.
The models used in our experiments and the dataset with limited expert predictions are available [here](https://drive.google.com/drive/folders/1nAUlxdOmwC6ZNtch3rxwKwNUVrYNmxkV). We also include the transformed input dataset in order to generate synthetic experts with our framework. 

* ### Step 4: Generating the expert prediction table and capacity constraint tables

To place all the necessary data in the correct directories, and generate the synthetic expert prediction table utilized in our experiments, as well as the capacity constraint tables used in our benchmarks, the user needs to run "[load\_data.py](load_data.py)". The script only requires the user to specify the directory of the datasets downloaded in Step 3.

## Replicating our experiments

### L2D Baseline Results
After following the steps to obtain the **Fraud Detection Dataset**, detailed in the previous section, the user must run the file "[run_tests.py](asfasf)". This script produces the test split assignments for each testing scenario generated in Step 4 of the previous section. These assignments are obtained by using each of our 3 baseline models, detailed in Section 4.2 of the [paper](sadfa),  resulting in a total of 660 sets of assignments. For details on the total compute time necessary to run all experiments, consult Table 7, in Section C.1 of the [appendix](sdfa). 

### ML Model and Human Expertise Model evaluation

The plots, numerical results, and hyperparameter choices relating to our ML model, detailed in Section B.1 of the [appendix](sdfa), are obtained using the script [ml_model/training_and_predicting.py](asdf). 

The plots, numerical results, and hyperparameter choices relating to our Human Expertise model, detailed in Section B.3, are obtained using the notebook [expertise_models/model_analysis.ipynb](asdf). 

### Synthetic expert's decision evaluation

The plots and numerical results regarding our synthetic expert's generation process and decision properties are obtained using the notebook [experts/expert_properties.ipynb](asdf). 

## Using the OpenL2D Framework

### Generating Synthetic Expert Decisions
To generate synthetic expert decisions, a user can define the necessary parameters in the file [experts/cfg.yaml](experts/config.yaml). For more details on each parameter and the decision generation process, consult Section 3.2 of the [paper](sadfa). Then, the user needs to run the script [experts/expert_gen.py](..). This script produces the decision table as well as information regarding the expert decision generation properties. These include the sampled parameters for each expert, the probabilities of error of each expert for each instance, and other useful information. 

To analyze your generated synthetic expert decisions, and tweak the expert properties according to your needs, consider using the notebook [experts/expert_properties.ipynb](xcv).

### Generating Datasets with limited human predictions

To generate one or more datasets with limited human predictions, the user needs to define the capacity constraints of each desired dataset, in the file [testbed/cfg.yaml](sdf), and run the script [testbed/testbed_train_generation.py](asdf). 

For each desired dataset, this script creates a subfolder within the folder [testbed/train]. Each dataset's subfolder contains that dataset's capacity constraints tables ("batches.parquet" and "capacity.parquet") and the dataset with limited expert predictions ("train.parquet").

### Generating test scenarios with different capacity constraints

To generate a set of capacity constraints to be applied in testing, the user needs to define the capacity constraints of each scenario, in the file [testbed/cfg.yaml](sdf), and run the script [testbed/testbed_test_generation.py](asdf). For each of the defined test scenarios, the script creates a subfolder within [testbed/test](adf). This subfolder contains the capacity constraint tables ("batches.csv" and "capacity.csv") to be used in testing.

### Running your own L2D testing

To run your own experiments within your generated tested scenarios, OpenL2D currently supports the use of the baselines described in Section 4.2 of the [paper](asdf). The user may define the experiments' parameters in the file [testbed/cfg.yaml](sdf), and then run the script [testbed/run\_tests.py](asdf).




