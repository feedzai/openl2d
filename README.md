# OpenL2D: A Benchmarking Framework for Learning to Defer in Human-AI Decision-Making

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

* Instructions and code necessary to:
  * Use the OpenL2D synthetic data generation framework.
  * Generate the FiFAR dataset, available [here](https://drive.google.com/file/d/1ZHleGXqi3Oxu-gmvRnKEsiBXjjAMAdi4/view?usp=sharing).
  * Conduct our L2D benchmarks.
* Notebooks for evaluation of FiFAR experts' properties and L2D benchmarks.

The submitted version of the paper is available [here](Documents/Paper.pdf).

To ensure complete reproducibility, all the models, data (generated or otherwise), and results mentioned in our paper are included in the FiFAR dataset.

## Creating the Python Environment
Requirements:
* anaconda3
  
After cloning the repo, please create and activate the Python environment by running

```
conda env create -f openl2d-environment.yml
conda activate openl2d-env
```

## Using FiFAR

The FiFAR dataset is publicly available [here](https://drive.google.com/file/d/1ZHleGXqi3Oxu-gmvRnKEsiBXjjAMAdi4/view?usp=sharing). 
This dataset includes

* Version 1 of the Base Variant of the Bank Account Fraud Dataset.
* Alerts raised by the Alert Model, accompanied by said model's score.
* Synthetically generated predictions from a team of 50 fraud analysts.
* 25 distinct training scenarios, as well as 5 distinct testing scenarios for each trained algorithm.
* ML models used in the data generation process, technical validation, and L2D benchmarking.
  * Alert Model.
  * Feature Dependence Models.
  * DeCCaF and OvA Models - Classifier *h*, OvA Classifiers, and DeCCaF's team correctness prediction models.

This dataset is more thouroughly described in Section 4 of the [Data Descriptor](Documents/Paper.pdf).

We encourage researchers to use FiFAR in order to develop L2D methods under realistic conditions. Our dataset poses realistic challenges, such as

* Limited expert prediction availability - only one prediction per expert.
* Dynamic environment - subject to label and concept shift.
* Human work capacity constraints.

We also facilitate further analysis of our generated experts and the conducted benchmarks, by providing users with two Jupyter Notebooks

* results.ipynb - which contains
  * evaluation of the deferral performance of all considered L2D baselines
  * evaluation of the performance and calibration of Classifier *h*, OvA Classifiers, and DeCCaF's team correctness prediction models.
* expert_analysis.ipynb - which contains the evaluation of the expert decision-making process properties (intra and inter-rater agreement, feature dependence, fairness and performance) 

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




