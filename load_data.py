import pandas as pd
import shutil
import subprocess
import yaml
import os
dataset_models_path = '/mnt/home/jean.alves/datasets_and_models'

#Copy Datasets and Models to the main code structure

#Preprocessed Expert Data
if not os.path.isdir('./experts/transformed_data'):   
    shutil.copytree(dataset_models_path + '/experts/transformed_data','./experts/transformed_data')

#ML Model
if not os.path.isdir('./ml_model/model'):  
    shutil.copytree(dataset_models_path + '/ml_model/model', './ml_model/model')

#Expertise Model
if not os.path.isdir('./expertise_models/models/small_regular/human_expertise_model'):
    shutil.copytree(dataset_models_path + '/expertise_models/models/small_regular/human_expertise_model','./expertise_models/models/small_regular/human_expertise_model')

#Dataset with limited expert predictions
if not os.path.isdir('./testbed/train/small#regular'):
    shutil.copytree(dataset_models_path + '/testbed/train/small#regular', './testbed/train/small#regular')

Input_Data = pd.read_csv('./data/Base.csv')

Input_Data.sort_values(by = 'month', inplace = True)
Input_Data.reset_index(inplace=True)
Input_Data.drop(columns = 'index', inplace = True)
Input_Data.index.rename('case_id', inplace=True)

data_cfg_path = './data/dataset_cfg.yaml'
with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

Input_Data.loc[:,data_cfg['data_cols']['categorical']] = Input_Data.loc[:,data_cfg['data_cols']['categorical']].astype('category')

Input_Data.to_parquet('./data/BAF.parquet')

subprocess.run(["python", "./ml_model/training_and_predicting.py"])
subprocess.run(["python", "./experts/expert_gen.py"])
subprocess.run(["python", "./testbed/testbed_test_generation.py"])






