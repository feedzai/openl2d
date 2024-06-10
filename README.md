# OpenL2D: A Benchmarking Framework for Learning to Defer in Human-AI Decision-Making

## Abstract

Learning to Defer (L2D) algorithms aim to improve human-AI collaboration in decision-making by deferring decisions to human experts when they are more likely to be correct than a model. The development of these systems is primarily hindered by the high cost of obtaining expert predictions for training and evaluation, often leading researchers to consider simplistic simulated expert behavior in their benchmarks. To address this, we introduce OpenL2D, a novel framework that generates realistic synthetic experts with adjustable bias and feature dependence. OpenL2D also creates expert work capacity constraints, limiting deferrals to experts within a given time window, thus allowing researchers to test L2D in realistic scenarios. We apply OpenL2D to a public fraud detection dataset to create the financial fraud alert review dataset (FiFAR), containing predictions from 50 fraud analysts for 30K alerted instances. We validate FiFAR's synthetic experts based on decision-making literature and their similarity to real fraud analysts. Finally, we benchmark L2D baselines under diverse conditions to emphasize the importance of considering complex expert decision-making processes in L2D.

## Overview

* [Resources](#Resources)
* [Using FiFAR](#Using-FiFAR)
* [Replicating the Data Generation Process and L2D Benchmarking](#Replicating-the-Data-Generation-Process-and-L2D-Benchmarking)
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

### Creating the Python Environment

Requirements:
* anaconda3
  
Before using any of the provided code, to ensure reproducibility, please create and activate the Python environment by running

```
conda env create -f openl2d-environment.yml
conda activate openl2d-env
```

## Using FiFAR

The FiFAR dataset is publicly available [here (pending figshare link)]().

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

We also facilitate further analysis of our generated experts and the conducted benchmarks, by providing users with two Jupyter Notebooks
* [Code/deferral/results.ipynb](Code/deferral/results.ipynb) - which contains
  * evaluation of the deferral performance of all considered L2D baselines
  * evaluation of the performance and calibration of Classifier *h*, OvA Classifiers, and DeCCaF's team correctness prediction models.
* [Code/synthetic_experts/expert_analysis.ipynb](Code/synthetic_experts/expert_analysis.ipynb) - which contains the evaluation of the expert decision-making process properties (intra and inter-rater agreement, feature dependence, fairness and performance) 

To use our code on FiFAR out-of-the-box, note that you should place it within the Repo's directory as such:

```
openl2d
│   README.md
│   .gitignore  
│   openl2d-environment.yml
│
└─── Code
│   │   ...
│   
└─── FiFAR
    │   ...
│   
└─── OpenL2D Use Example
    │   ...
```


## Replicating the Data Generation Process and L2D Benchmarking

To replicate the generation of FiFAR, as well as our experiments, please execute the following steps:

**Attention**: Run each python script **inside** the folder where it is located, to ensure the relative paths within each script work correctly

**Note**: Should you wish to only replicate a few of the steps, ensure that said step's output is deleted (i.e., delete the trained alert model if you wish to retrain it).

### Step 1 - Clone the Repo and Download the Base Dataset

If you wish to replicate the data generation process and L2D Benchmarking, you must download either FiFAR or the [Base.csv](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/versions/1?select=Base.csv) file of the BAF dataset.

After cloning the repo, please place a new directory named "FiFAR" inside the repo's folder, ensuring that your directory looks like this. Note that the FiFAR directory should only contain a folder named "alert_data" with the Base.csv file within it

```
openl2d
│   README.md
│   .gitignore  
│   openl2d-environment.yml
│
└─── Code
│   │   ...
│   
└─── FiFAR
    │   
    └─── alert_data
         │   
         └─── Base.csv
│   
└─── OpenL2D Use Example
    │   ...
```

### Step 2 - Activate the Environment
To activate the Python environment with the necessary dependencies please follow [these steps](#Creating-the-Python-Environment)

### Step 3 - Train the Alert Model and create the set of alerts
To train the Alert Model, run the file [Code/alert_model/training_and_predicting.py](Code/alert_model/training_and_predicting.py), which will train the Alert Model and score all instances in months 4-8 of the BAF dataset.

Then, run the file [Code/alert_data/preprocess.py](Code/alert_data/preprocess.py), to create the dataset of 30K alerts raised in months 4-8. This will be the set of instances used over all following generation processes.

### Step 4 - Generate the Synthetic Expert predictions
To generate all the data within the folder "synthetic_experts" of FiFAR, run the script [Code/synthetic_experts/expert_gen.py](Code/synthetic_experts/expert_gen.py), which will generate the synthetic expert predictions, and also save their sampled parameters, calculated probabilities of error for each alerted instance, as well as the list of expert id's.

### Step 5 - Generate the Training and Testing Scenarios
To generate all 25 training scenarios, run the script [Code/testbed/testbed_train_alert_generation.py](Code/testbed/testbed_train_alert_generation.py).
To generate the 5 distinct capacity constraints to be applied to each of the deferral methods in testing, run the script [Code/testbed/testbed_test_generation.py](Code/testbed/testbed_test_generation.py).

### Step 6 - Train OvA and DeCCaF algorithms
As both of these algorithms share the classifier *h* (see Section *Training OvA and DeCCaF Baselines* of the [paper](Documents/Paper.pdf)), we first train this classifier, by running the script [Code/classifier_h/training.py](Code/classifier_h/training.py).

To train the OvA Classifiers run [Code/expert_models/run_ova.py](Code/expert_models/run_ova.py). To train the DeCCaF classifiers run [Code/expert_models/run_deccaf.py](Code/expert_models/run_deccaf.py)

### Step 7 - Run the Deferral Experiments

To reproduce the deferral testing run the script [Code/deferral/run_alert.py](Code/deferral/run_alert.py). These results can then be evaluated with the notebook [Code/deferral/results.ipynb](Code/deferral/results.ipynb)


## Using the OpenL2D Framework

![alt text](Images/framework_diagram.png)

### Defining the Input Dataset Properties
To use OpenL2D to generate experts on any tabular dataset, the file [Code/alert_data/dataset_cfg.yaml](Code/alert_data/dataset_cfg.yaml) must be adapted to your particular needs. 

This involves:
#### 1. Defining the dataset's columns.

* The user **must** define which column corresponds to the ground truth label
* The categorical features **must** also be defined if they exist.

Optionally the user may also define:
* The timestamp column: which can be used posteriorly to define training and testing splits, and can be taken into account in the generation of capacity constraints.
* The protected attribute column: which can be used to simulate experts with bias against a particular group based on said attribute.
* The model_score column: which can be used to simulate experts who have access to exterior information such as an ML Model's score.

FiFAR Example:
```yaml
data_cols:
  label: 'fraud_bool'       
  timestamp: 'month'         
  protected: 'customer_age' 
  model_score: 'model_score' 
  categorical:             
    - "payment_type"
    - "employment_status"
    - "housing_status"
    - "source"
    - "device_os"
```

#### 2. Defining the categorical dictionary.

For each categorical feature, the user must define the dictionary of possible categorical values, by setting
 * Key: categorical feature's column
 * Values: all possible values for said feature

This is done to ensure that the categorical features are encoded identically when passed to the LGBM models.

FiFAR Example:
```yaml
categorical_dict:
  device_os:
  - linux
  - macintosh
  - other
  - windows
  - x11
  employment_status:
  - CA
  - CB
  - CC
  - CD
  - CE
  - CF
  - CG
  housing_status:
  - BA
  - BB
  - BC
  - BD
  - BE
  - BF
  - BG
  payment_type:
  - AA
  - AB
  - AC
  - AD
  - AE
  source:
  - INTERNET
  - TELEAPP
```

#### 3. Defining the cost structure of the classification task
The user must also define the cost structure of the problem, by setting lambda = (cost of a false positive)/(cost of a false negative)
This value will be used in case the user wishes to sample expert performance as measured by the misclassification cost.

FiFAR example:
```yaml
lambda: 0.057
```

### Generating Synthetic Expert Decisions
To generate synthetic expert decisions, a user must place the following scripts in a folder:

* [Code/synthetic_experts/expert_gen.py](Code/synthetic_experts/expert_gen.py) - responsible for sampling the expert parameters and generating the expert objects
* [Code/synthetic_experts/expert_src.py](Code/synthetic_experts/expert_src.py) - contains the source code for the expert objects
* [Code/synthetic_experts/cfg.yaml](Code/synthetic_experts/cfg.yaml) - contains the user defined configurations to generate synthetic experts. This file also contains a detailed description of the necessary user inputs

The user then only needs to define the necessary parameters in the "cfg.yaml" file, as such:

#### 1. Defining the input and output paths

The user must define the following paths, relative to the location of the "expert_gen.py" script:
* data_cfg_path: Path to the previously defined "dataset_cfg.yaml"
* dataset_path: Path to the data on which to generate synthetic expert decisions
* destination_path: Output path for the generated expert decisions and sampled expert parameters     

FiFAR Example:
```yaml
data_cfg_path: '../alert_data/dataset_cfg.yaml'                       
dataset_path: '../../FiFAR/alert_data/processed_data/alerts.parquet' 
destination_path: '../../FiFAR/synthetic_experts'                         
```
#### 2. Defining the partition on which to fit the expert's performance
The user must define which partition of the dataset should be used to fit the values of beta_0 and beta_1
* **Option 1** - If the dataset has a timestamp column, fitting_set should be defined as the dates that delimit the partition
* **Option 2** - If the dataset does not have a timestamp column, fitting_set should be defined as the indexes that delimit the partition
The intervals are defined as [start,end) - the last value is not included

FiFAR Example:
```yaml
fitting_set: [6,7] #We want to use the totality of month 6
```

#### 3. Defining the properties related to the protected attribute
*Note* - These can be ommited if there is no protected attribute

* **Option 1** - If your protected attribute is NUMERICAL, you must define:
  * protected_threshold - value that separates the two distinct groups
  * protected_values - whether the values 'higher' or 'lower' than the parameter are the protected group.
* **Option 2** -If your protected attribute is CATEGORICAL, you must define:
  * protected_class - value that corresponds to the protected group.

FiFAR Example:
```yaml
protected_threshold: 50
protected_values: 'higher'
```

#### 3. Defining the expert group properties
The experts are defined under the 'experts' key in the "cfg.yaml" File.

For a given group, the user **must** first set:
* n - The number of experts belonging to said group
* group_seed - The random seed used in the sampling processes involved in the expert generation.

The user must then define the feature weight sampling process (see Section *Synthetic Data Generation Framework - OpenL2D* of the Paper):
 * **Option 1** - Define each individual feature's weight distribution. This involves defining:
    * w_dict: containing one (mean, stdev) pair per feature in your dataset. These include the protected attribute and model score if they exist.
    * *Example* - see file [Code/synthetic_experts/cfg.yaml](Code/synthetic_experts/cfg.yaml)
 * **Option 2** - Define the parameters for a spike and slab distribution:
    * w_mean: mean of the gaussian slab distribution
    * w_stdev: standard deviation of the gaussian slab distribution 
    * theta: probability that weight is sampled from the slab distribution.
    * You can optionally also define the weights for the protected attribute and the model_score separately, by defining
      *  protected_mean
      *  protected_stdev
      *  score_mean
      *  score_stdev
    * *Example* - see file [OpenL2D_Use_Example/synthetic_experts/cfg.yaml](OpenL2D_Use_Example/synthetic_experts/cfg.yaml)  

For both options, the user must also define the distribution of the *alpha* parameter, by setting:

* alpha_mean: mean of the gaussian distribution from which alpha is sampled
* alpha_stdev: standard deviation of the gaussian distribution from which alpha is sampled

The user **must** then define the performance distribution of the expert group:
 * **Option 1** - Defining the cost distribution, by setting:
    * target_mean: Mean of the gaussian distribution from which the expert's expected misclassification cost is sampled
    * target_stdev: Standard deviation of the gaussian distribution from which the expert's expected misclassification cost is sampled
    * OPTIONAL - top_clip: Maximum value that can be sampled for the cost
        * If not defined, there is no upper bound on the cost
    * OPTIONAL - bottom_clip: Minimum value that can be sampled for the cost
        * If not defined, the lower bound for the cost will be set to 0
    * *Example* - see file [Code/synthetic_experts/cfg.yaml](Code/synthetic_experts/cfg.yaml)
  
 * **Option 2** - Defining the fpr and fnr distributions, by setting, for each:
    * target_mean: Mean of the gaussian distribution from which the expert's fpr/fnr is sampled
    * target_stdev: Standard deviation of the gaussian distribution from which the expert's fpr/fnr is sampled
    * *Example* - see file [OpenL2D_Use_Example/synthetic_experts/cfg.yaml](OpenL2D_Use_Example/synthetic_experts/cfg.yaml)

 Optionally, the user may also define, for both of the aforementioned options:
 * max_FPR - if not defined, the upper bound is set to 1
 * min_FPR - if not defined, the lower bound is set to 0
 * max_FNR - if not defined, the upper bound is set to 1
 * min_FNR - if not defined, the lower bound is set to 0

More expert groups may be defined under the 'experts' key. First, the user needs to set the 'baseline_group'.
In subsequent expert groups, only the parameters that differ from the baseline group need to be defined.
*NOTE*: If a subsequent group uses a parameter that was not defined in the baseline group (i.e. theta), it will not be recognized. In this case, users must set theta = 1 (default value) on the baseline group, and then they may define the subsequent group's theta value.

For more details on each parameter and the decision generation process, consult Section *Synthetic Data Generation Framework - OpenL2D* of the [paper](Documents/Paper.pdf).

The user then only needs to run the script [Code/synthetic_experts/expert_gen.py](Code/synthetic_experts/expert_gen.py). This script produces the decision table as well as information regarding the expert decision generation properties (see Section 4 of the [paper](Documents/Paper.pdf)).


### Generating Training and Testing Scenarios

To generate these scenarios, a user must place the following scripts in a folder:

* [Code/testbed/testbed_train_alert_generation.py](Code/testbed/testbed_train_alert_generation.py)- responsible for generating training scenarios
* [Code/testbed/testbed_test_generation.py](Code/testbed/testbed_test_generation.py) - responsible for generating testing scenarios
* [Code/testbed/cfg.yaml](Code/testbed/cfg.yaml) - contains the user defined configurations to generate the testbed. This file also contains a detailed description of the necessary user inputs

The user then only needs to define the necessary parameters in the "cfg.yaml" file, as such:

#### 1. Defining the input and output paths
The user must first define the paths pertaining to the dataset and generated expert predictions, as well as the output paths for the testing and training scenarios:
* dataset_path: Path to the dataset on which expert decisions were generated
* data_cfg_path: Said dataset's config file
* expert_folder_path: Path containing the outputs from expert_gen.py
* destination_path_train: Output directory of the generated training scenarios
* destination_path_test: Output directory of the generated test scenarios

FiFAR Example:

```yaml
dataset_path: '../../FiFAR/alert_data/processed_data/alerts.parquet'
data_cfg_path: '../alert_data/dataset_cfg.yaml' 
expert_folder_path:  '../../FiFAR/synthetic_experts'
destination_path_train: '../../FiFAR/testbed/train_alert'
destination_path_test: '../../FiFAR/testbed/test' 
```

#### 2. Setting the random seed, training and testing splits, and whether there are timestamp constraints 

The user can set the random seed for the generation of the testing and training scenarios, by defining 'random_seed'

The user must then define which partitions of the dataset should be used to generate the training and test scenarios, by defining
* training_set
* test_set
This can be done in one of two ways, depending on whether the dataset contains a timestamp column.
* **Option 1** - If the dataset has a timestamp column, fitting_set should be defined as the dates that delimit the partition
* **Option 2** - If the dataset does not have a timestamp column, fitting_set should be defined as the indexes that delimit the partition
The intervals are defined as [start,end) - the last value is not included

If there is a time related restriction on how batches must be distributed, the user may set 'timestamp_constraint' as True, ensuring that only instances with the same timestamp can belong to a batch. This is useful, for example, to generate batches which contain instances pertaining to the same day.

FiFAR Example:

```yaml
random_seed: 42
training_set: [3,7] 
test_set: [7,8]

#in our experiments, a batch can only contain instances belonging to the same month
timestamp_constraint: True
```


#### 3. Setting the capacity constraints for the training and testing scenarios

The user must define at least one batch and one capacity configuration for training and testing. 
Each batch configuration will be combined with each capacity configuration to generate an array of scenarios.

To define the batch vector, the user must set:
* 'size' - Number of instances within a batch
* 'seed' - Seed used to distribute cases throughout batches.

To define the capacity matrix, the user must set:
* 'deferral_rate' - Fraction of instances that can be deferred to humans
* 'distribution' - {'variable','homogeneous'} - Distribution of the work capacity of the expert team

If the distribution is variable, the user must set:
* 'distribution_stdev' - standard deviation of the gaussian distribution from which the expert's capacity constraints are sampled. This gaussian distribution's mean is 'deferral_rate'*'size'/'n_experts', that is, the value corresponding to an homogeneous distribution. The standard deviation is calculated as 'distribution_stdev' * 'deferral_rate' * 'size'/'n_experts'.
* 'distribution_seed' - seed for the individual expert capacity sampling
* 'variable_capacity_per_batch' - {True,False} - whether or not the same expert capacity is generated for all batches

The user may also set the value of 'n_experts', limiting the number of experts available in each batch.
If 'n_experts' is set, the user must set
* 'n_experts_seed' - seed for the sampling of which experts are available
* 'variable experts_per_batch' - {True, False} - whether or not the experts available per batch can be different.

A simple example follows, and is available in [OpenL2D_Use_Example/testbed/cfg.yaml](OpenL2D_Use_Example/testbed/cfg.yaml):

```yaml
environments_train:
  batch:
    shuffle_1:
      size: 500
      seed: 42
  capacity:
    team_1:
      deferral_rate: 1
      distribution: 'homogeneous'

environments_test:
  batch:
    shuffle-1:
      size: 500
      seed: 42
  capacity:
    team_1-hom:
      deferral_rate: 0.8
      n_experts: 5
      n_experts_seed: 42
      variable_experts_per_batch: True
      distribution: 'homogeneous'
```

For an example with more batch and capacity configurations in testing and in training, see [Code/testbed/cfg.yaml](Code/testbed/cfg.yaml).

Then, the user may run the script 'testbed_train_alert_generation.py'. For each desired scenario, this script creates a subfolder within the defined output path. Each dataset's subfolder contains that training scenarios's capacity constraints tables ("batches.csv" and "capacity.csv") and the dataset with limited expert predictions ("train.parquet").

To generate a set of capacity constraints to be applied in testing, the user needs to run the script 'testbed_test_generation.py'. For each of the defined test scenarios, the script creates a subfolder within within the defined output path. This subfolder contains the capacity constraint tables ("batches.csv" and "capacity.csv") to be used in testing.


