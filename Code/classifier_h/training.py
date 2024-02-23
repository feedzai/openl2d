# %%
import pandas as pd
import yaml
import numpy as np
import hpo_wce
import os
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

def sig(x):
    return 1/(1+np.exp(-x))

def output(data, model, init_score):
    return sig(model.predict(data,raw_score=True) + init_score)

with open('../alert_data/dataset_cfg.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)

n_jobs = 10

# DATA LOADING ---------------------------------------------------------------------------------
data = pd.read_parquet(f'../../Dataset/alert_data/processed_data/alerts.parquet')
data = cat_checker(data, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

train = data.loc[(data["month"] > 2) & (data["month"] < 6)]
val = data.loc[data["month"] == 6]

X_train = train.drop(columns = ['fraud_bool','model_score','month'])
y_train = train['fraud_bool']

X_val = val.drop(columns = ['fraud_bool','model_score','month']) 
y_val = val['fraud_bool']

w_train = y_train.replace([0,1],[data_cfg['l'],1])
w_val = y_val.replace([0,1],[data_cfg['l'],1])

p_train = (y_train*w_train).sum()/(w_train.sum())
p_val = (y_val*w_val).sum()/(w_val.sum())

init_train = np.log((p_train)/(1-p_train))
init_val = np.log((p_val)/(1-p_val))

n = 0
for param_space_dic in os.listdir('./param_spaces/'):
    with open('./param_spaces/' + param_space_dic, 'r') as infile:
        param_space = yaml.safe_load(infile)

    for initial in np.arange(init_train, init_train + 2, 0.2):
        param_space['init_score'] = initial
        os.makedirs(f'../../Dataset/classifier_h/models/', exist_ok=True)
        
        if not (os.path.exists(f'../../Dataset/classifier_h/models/model_{n}')):
            opt = hpo_wce.HPO(X_train,X_val,y_train,y_val,w_train,w_val, parameters = param_space, method = 'TPE', path = f"../../Dataset/classifier_h/models/model_{n}")
            opt.initialize_optimizer(data_cfg['categorical_dict'], n_jobs)
            n +=1
        else:
            print('model is trained')
            n +=1

Trials = []

for model in os.listdir('../../Dataset/classifier_h/models/'):
    study = int(model.split('_')[-1])
    with open('../../Dataset/classifier_h/models/' + model + '/history.yaml', 'r') as infile:
        param_hist = yaml.safe_load(infile)

    with open('../../Dataset/classifier_h/models/' + model + '/config.yaml', 'r') as infile:
        conf = yaml.safe_load(infile)
    
    temp = pd.DataFrame(param_hist)
    temp['study'] = study
    temp['max_depth_max'] = conf['params']['max_depth']['range'][1]
    Trials.append(temp)

Trials = pd.concat(Trials)
Trials = Trials.reset_index(drop = True)
Trials['study'] = Trials['study'].astype(int)
a = Trials

selec_ix = a.loc[a['ll'] == a['ll'].min(),'study'].to_numpy()[0]

selected_model_path = f'../../Dataset/classifier_h/models/model_{selec_ix}'

with open(f'{selected_model_path}/best_model.pickle', 'rb') as infile:
    model = pickle.load(infile)

with open(f'{selected_model_path}/config.yaml', 'r') as infile:
    model_cfg = yaml.safe_load(infile)

test = data.loc[data["month"] == 7]

X_test = test.drop(columns = ["month",'model_score', "fraud_bool"])
y_test = test["fraud_bool"]
w_test = y_test.replace([0,1],[data_cfg['l'],1])

p_test = (y_test*w_test).sum()/(w_test.sum())
init_test = np.log((p_test)/(1-p_test))

selected_model = dict()
init_score = model_cfg['init_score']
selected_model['init_score'] = float(init_score)
selected_model['threshold'] = 0.5

model_preds = pd.Series(output(X_train,model, init_score) >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_train, model_preds).ravel()
avg_cost_model = (data_cfg['l']*fp + fn)/(tn+fp+fn+tp)

selected_model['fpr_train'] = float(fp/(fp+tn))
selected_model['fnr_train'] = float(fn/(fn+tp))
selected_model['prev_train'] = float(y_train.mean())
selected_model['cost_train'] = float(avg_cost_model)

tn, fp, fn, tp = confusion_matrix(y_train, np.ones(len(y_train))).ravel()
avg_cost_full_rej = (data_cfg['l']*fp + fn)/(tn+fp+fn+tp)

print(f"Training Set -- Model: {avg_cost_model:.3f}. Rejecting all: {avg_cost_full_rej:.3f}")

model_preds = pd.Series(output(X_val,model, init_score) >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_val, model_preds).ravel()
avg_cost_model = (data_cfg['l']*fp + fn)/(tn+fp+fn+tp)

selected_model['fpr_val'] = float(fp/(fp+tn))
selected_model['fnr_val'] = float(fn/(fn+tp))
selected_model['prev_val'] = float(y_val.mean())
selected_model['cost_val'] = float(avg_cost_model)

tn, fp, fn, tp = confusion_matrix(y_val, np.ones(len(y_val))).ravel()
avg_cost_full_rej = (data_cfg['l']*fp + fn)/(tn+fp+fn+tp)

print(f"Val Set -- Model: {avg_cost_model:.5f}. Rejecting all: {avg_cost_full_rej:.5f}")

model_preds = pd.Series(output(X_test,model, init_score) >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, model_preds).ravel()
avg_cost_model = (data_cfg['l']*fp + fn)/(tn+fp+fn+tp)

selected_model['fpr_test'] = float(fp/(fp+tn))
selected_model['fnr_test'] = float(fn/(fn+tp))
selected_model['prev_test'] = float(y_test.mean())
selected_model['cost_test'] = float(avg_cost_model)

tn, fp, fn, tp = confusion_matrix(y_test, np.ones(len(y_test))).ravel()
avg_cost_full_rej = (data_cfg['l']*fp + fn)/(tn+fp+fn+tp)

print(f"Test Set -- Model: {avg_cost_model:.5f}. Rejecting all: {avg_cost_full_rej:.5f}")

os.makedirs(f'../../Dataset/classifier_h/selected_model/', exist_ok=True)

with open(f'../../Dataset/classifier_h/selected_model/best_model.pickle', 'wb') as outfile:
    pickle.dump(model, outfile)

with open(f'../../Dataset/classifier_h/selected_model/model_properties.yaml', 'w') as outfile:
    yaml.dump(selected_model, outfile)