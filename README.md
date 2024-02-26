﻿# OpenL2D: A Benchmarking Framework for Learning to Defer in Human-AI Decision-Making

## Abstract

Public dataset limitations have significantly hindered the development and benchmarking of _learning to defer_ (L2D) algorithms, which aim to optimally combine human and AI capabilities in hybrid decision-making systems. The development of these systems is primarily hindered by the high cost of obtaining human predictions for training and evaluation, leading researchers to often consider simplistic simulated human behaviour in their benchmarks. To overcome this challenge, we introduce OpenL2D, a novel framework designed to generate synthetic expert decisions and testbed settings for L2D methods. OpenL2D facilitates the creation of synthetic experts with adjustable bias and feature dependence, simulates realistic human work capacity constraints, and provides diverse training and testing conditions. We employ OpenL2D on a public fraud detection dataset to generate the _Financial Fraud Alert Review Dataset_ (FiFAR), containing predictions from a team of 50 fraud analysts for 30K alerted instances. We benchmark L2D baselines under a diverse array of conditions, subject to expert capacity constraints, demonstrating the unique, real-world challenges posed by FiFAR relative to previous benchmarks.

## Overview

* [Resources](#Resources)
* [Installing Necessary Dependencies](#Installing-Necessary-Dependencies)
* [Using the OpenL2D Fraud Detection Dataset](#Using-the-OpenL2D-Fraud-Detection-Dataset)
* [Replicating our Experiments](#Replicating-our-experiments)
* [Using the OpenL2D Framework](#Using-the-OpenL2D-Framework)

## Resources
In this repo, we provide users with:

* Code necessary to:
  * Use the OpenL2D synthetic data generation framework.
  * Generate the FiFAR dataset.
  * Conduct our L2D benchmarks.
* Notebooks for evaluation of FiFAR experts' properties and L2D benchmarks.

The submitted version of the paper, the appendix, and the Datasheet are available in the following links:

* [Paper](Documents/Paper.pdf)
* [Appendix](Documents/Appendix.pdf)
* [Datasheet](Documents/Datasheet.pdf)

## Installing Necessary Dependencies

To use the provided code, please install the package available in the folder [Dependencies](Dependencies).

## Using the OpenL2D Fraud Detection Dataset

![alt text](Images/dataset_diagram.png)

In our experiments, for training and testing of assignment methods, we utilized the **OpenL2D Fraud Detection Dataset**. This dataset is comprised of:

* An Input Dataset.
* Synthetic Expert prediction table.
* Dataset with limited expert predictions.
* Sets of capacity constraint tables.

For more information on each of these components, please consult the provided [Datasheet](Documents/Datasheet.pdf).

* ### Step 1: Download the Code in this repo:
The sets of capacity constraint tables and the synthetic expert prediction table are generated by using the input dataset and a set of configs included in this repo. For easy use of our dataset and available notebooks, we encourage users to download the repo in its entirety.

* ### Step 2: Download the Input Dataset
Our input dataset is the base variant of the Bank Account Fraud Tabular Dataset, available [here](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022?resource=download&select=Base.csv). This dataset should then be placed in the folder [OpenL2D/data](OpenL2D/data).

* ### Step 3: Download the Models, Dataset with limited expert predictions and other necessary data.
The models used in our experiments and the dataset with limited expert predictions are available [here](https://drive.google.com/drive/folders/1nAUlxdOmwC6ZNtch3rxwKwNUVrYNmxkV). We also include the transformed input dataset in order to generate synthetic experts with our framework. 

* ### Step 4: Generating the expert prediction table and capacity constraint tables

To place all the necessary data in the correct directories, and generate the synthetic expert prediction table utilized in our experiments, as well as the capacity constraint tables used in our benchmarks, the user needs to run "[load\_data.py](load_data.py)". The script only requires the user to specify the directory of the datasets downloaded in Step 3. The expert prediction table is split according to the expert preprocessing and deployment splits. For more information consult Section 4.1 of the [paper](Documents/Paper.pdf)

### Uses of the L2D Fraud Detection Dataset

This dataset can be used to develop L2D methods under realistic conditions. Our dataset poses realistic challenges, such as:

* Limited expert prediction availability
* Developing algorithms under dynamic environments
* Human work capacity constraints

The Dataset with limited expert predictions can be used to train assignment systemds under realistic human data availability. Our expert prediction table contains 50 synthetic fraud analyst's predictions for each of the 1M instances of the BAF dataset. It can be used to train more data demanding algorithms, or to generate different training scenarios with the use of new capacity constraints. Our capacity constraint tables are also available, and are useful to test capacity aware assignment under a vast array of expert team configurations.



## Replicating our experiments

### L2D Baseline Results
After following the steps to obtain the **Fraud Detection Dataset**, detailed in the previous section, the user must run the file "[OpenL2D/testbed/run_tests.py](OpenL2D/testbed/run_tests.py)". This script produces the test split assignments for each testing scenario generated in Step 4 of the previous section. These assignments are obtained by using each of our 3 baseline models, detailed in Section 4.2 of the [paper](Documents/Paper.pdf),  resulting in a total of 660 sets of assignments. For details on the total compute time necessary to run all experiments, consult Table 7, in Section C.1 of the [appendix](Documents/Appendix.pdf). 

### ML Model and Human Expertise Model evaluation

The plots, numerical results, and hyperparameter choices relating to our ML model, detailed in Section B.1 of the [appendix](Documents/Appendix.pdf), are obtained using the script [OpenL2D/ml_model/training_and_predicting.py](OpenL2D/ml_model/training_and_predicting.py). 

The plots, numerical results, and hyperparameter choices relating to our Human Expertise model, detailed in Section B.3, are obtained using the notebook [OpenL2D/expertise_models/model_analysis.ipynb](OpenL2D/expertise_models/model_analysis.ipynb). 

### Synthetic expert's decision evaluation

The plots and numerical results regarding our synthetic expert's generation process and decision properties are obtained using the notebook [OpenL2D/experts/expert_properties.ipynb](OpenL2D/experts/expert_properties.ipynb). 

## Using the OpenL2D Framework

![alt text](Images/framework_diagram.png)

### Generating Synthetic Expert Decisions
To generate synthetic expert decisions, a user can define the necessary parameters in the file [OpenL2D/experts/cfg.yaml](OpenL2D/experts/cfg.yaml). For more details on each parameter and the decision generation process, consult Section 3.2 of the [paper](Documents/Paper.pdf). Then, the user needs to run the script [OpenL2D/experts/expert_gen.py](OpenL2D/experts/expert_gen.py). This script produces the decision table as well as information regarding the expert decision generation properties. These include the sampled parameters for each expert, the probabilities of error of each expert for each instance, and other useful information. 

To analyze your generated synthetic expert decisions, and tweak the expert properties according to your needs, consider using the notebook [OpenL2D/experts/expert_properties.ipynb](OpenL2D/experts/expert_properties.ipynb).

### Generating Datasets with limited human predictions

To generate one or more datasets with limited human predictions, the user needs to define the capacity constraints of each desired dataset, in the file [OpenL2D/testbed/cfg.yaml](OpenL2D/testbed/cfg.yaml), and run the script [OpenL2D/testbed/testbed_train_generation.py](OpenL2D/testbed/testbed_train_generation.py). 

For each desired dataset, this script creates a subfolder within a folder "OpenL2D/testbed/train". Each dataset's subfolder contains that dataset's capacity constraints tables ("batches.parquet" and "capacity.parquet") and the dataset with limited expert predictions ("train.parquet").

### Generating test scenarios with different capacity constraints

To generate a set of capacity constraints to be applied in testing, the user needs to define the capacity constraints of each scenario, in the file [OpenL2D/testbed/cfg.yaml](OpenL2D/testbed/cfg.yaml), and run the script [OpenL2D/testbed/testbed_test_generation.py](OpenL2D/testbed/testbed_test_generation.py). For each of the defined test scenarios, the script creates a subfolder within [OpenL2D/testbed/test](OpenL2D/testbed/test). This subfolder contains the capacity constraint tables ("batches.csv" and "capacity.csv") to be used in testing.

### Running your own L2D testing

To run your own experiments within your generated tested scenarios, OpenL2D currently supports the use of the baselines described in Section 4.2 of the [paper](Documents/Paper.pdf). The user may define the experiments' parameters in the file [OpenL2D/testbed/cfg.yaml](OpenL2D/testbed/cfg.yaml), and then run the script [OpenL2D/testbed/run\_tests.py](OpenL2D/testbed/run_tests.py).




