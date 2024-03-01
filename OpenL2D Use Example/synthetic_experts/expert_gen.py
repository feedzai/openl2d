# %%
import pandas as pd
import yaml
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler
from sklearn.metrics import confusion_matrix
import pickle
import expert_src as experts
import numpy as np
import os
from pathlib import Path
import shutil


cfg_path = 'cfg.yaml'

with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)

data_cfg_path =cfg['data_cfg_path']
dataset_path = cfg['dataset_path']
destination_path = cfg['destination_path']

class CustomException(Exception):
    pass

def sample_cost(mu,sigma,n,prevalence, l, top_clip, bottom_clip, max_FPR, min_FPR, max_FNR, min_FNR):
    slope = -(prevalence)/(l*(1-prevalence))
    costs = np.random.normal(loc = mu, scale = sigma, size = n)
    costs = np.clip(costs, bottom_clip, top_clip)
    experts = []

    for cost in costs:
        line = pd.DataFrame()
        line['x'] = np.random.uniform(0.0001,0.9999,size = 10000)
        line['y'] = line['x']*slope + cost/(l*(1-prevalence))
        line = line.loc[(line['y']>=min_FPR) & (line['y']<=max_FPR)]
        line = line.loc[(line['x']>=min_FNR) & (line['x']<=max_FNR)]
        selec = np.random.choice(line.index)
        experts.append([line['x'].loc[selec],line['y'].loc[selec]])
    
    experts = np.array(experts)

    return experts

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    return new_data


with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)


cat_dict = data_cfg['categorical_dict']
data = pd.read_parquet(dataset_path)

if 'lambda' in data_cfg:
    l = data_cfg['lambda']
else:
    l = 1

try:
    LABEL_COL = data_cfg['data_cols']['label']
except KeyError:
    print("Please define the label column in the dataset config file by using the key 'label' under 'data_cols'")
    raise

if 'categorical' in data_cfg['data_cols']:
    CATEGORICAL_COLS = data_cfg['data_cols']['categorical']
    data[CATEGORICAL_COLS] = data[CATEGORICAL_COLS].astype('category')
    data = cat_checker(data, CATEGORICAL_COLS, cat_dict)

try:
    fitting_set = cfg['fitting_set']
except KeyError:
    print("Please define the fitting_set in the file 'cfg.yaml'")
    raise

if 'timestamp' in data_cfg['data_cols']:
    TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
    non_fit_set = data.loc[~((data[TIMESTAMP_COL] >= fitting_set[0]) & (data[TIMESTAMP_COL] < fitting_set[1]))]
    fit_set = data.loc[((data[TIMESTAMP_COL] >= fitting_set[0]) & (data[TIMESTAMP_COL] < fitting_set[1]))]
    y_fit_set = fit_set[LABEL_COL]
else:
    fit_set = data.loc[fitting_set[0]:fitting_set[1],:]
    non_fit_set = data.drop(index = fit_set.index)
    y_fit_set = fit_set[LABEL_COL]
    TIMESTAMP_COL = None

if 'protected' in data_cfg['data_cols']:
    PROTECTED_COL = data_cfg['data_cols']['protected']
    if PROTECTED_COL in data_cfg['data_cols']['categorical']:
        protected_type = 'categorical'
        try:
            protected_class = cfg['protected_class']
        except KeyError:
            print("Please define the protected class in the file 'cfg.yaml'")
    else:
        protected_type = 'numerical'
        try:
            protected_threshold = cfg['protected_threshold']
        except KeyError:
            print("Please define the protected attribute's threshold in the file 'cfg.yaml'")
        try:
            protected_values = cfg['protected_values']
        except KeyError:
            print("Please define the protected attribute's values (higher or lower than threshold) in the file 'cfg.yaml'")
else:
    PROTECTED_COL = None

if 'model_score' in data_cfg['data_cols']:
    MLSCORE_COL = data_cfg['data_cols']['model_score']
else:
    MLSCORE_COL = None

try:
    LABEL_COL = data_cfg['data_cols']['label']
except KeyError:
    print("Please define the label column in the dataset config file by using the key 'label' under 'data_cols'")
    raise

try:
    baseline_group = cfg['baseline_group']
except KeyError:
    print("Please define the baseline_group in the file 'cfg.yaml'")
    raise

prevalence = y_fit_set.mean()
# Creating ExpertTeam object. 
expert_team = experts.ExpertTeam()
EXPERT_IDS = dict(human_ids=list())
#We use the ML Model training split to fit our experts.
#The expert fitting process involves determining the ideal Beta_0 and Beta_1 to obtain the user's desired target FPR and FNR
experts_fit_set_X = fit_set.copy().drop(columns=LABEL_COL)
experts_non_fit_set_X = non_fit_set.copy().drop(columns=LABEL_COL)

#Change customer_age variable to a binary
if PROTECTED_COL is not None:
    if protected_type == 'numerical':
        if protected_values == 'higher':
            experts_fit_set_X[PROTECTED_COL] = (experts_fit_set_X[PROTECTED_COL] >= protected_threshold).astype(int)
            experts_non_fit_set_X[PROTECTED_COL] = (experts_non_fit_set_X[PROTECTED_COL] >= protected_threshold).astype(int)
        if protected_values == 'lower':
            experts_fit_set_X[PROTECTED_COL] = (experts_fit_set_X[PROTECTED_COL] <= protected_threshold).astype(int)
            experts_non_fit_set_X[PROTECTED_COL] = (experts_non_fit_set_X[PROTECTED_COL] <= protected_threshold).astype(int)

    if protected_type == 'categorical':
        experts_fit_set_X[PROTECTED_COL] = (experts_fit_set_X[PROTECTED_COL] == protected_class).astype(int)
        experts_non_fit_set_X[PROTECTED_COL] = (experts_non_fit_set_X[PROTECTED_COL] == protected_class).astype(int)
    
    
#Transform the numerical columns into quantiles and subtract 0.5 so they exist in the [-0.5, 0.5] interval
cols_to_quantile = experts_fit_set_X.drop(columns=CATEGORICAL_COLS).columns.tolist()
qt = QuantileTransformer(random_state=42)
experts_fit_set_X[cols_to_quantile] = (
    qt.fit_transform(experts_fit_set_X[cols_to_quantile])
    - 0.5  # centered on 0
)
#Target encode and transform the categorical columns
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
experts_fit_set_X[CATEGORICAL_COLS] = oe.fit_transform(experts_fit_set_X[CATEGORICAL_COLS])
ss = StandardScaler(with_std=False)
experts_fit_set_X[:] = ss.fit_transform(experts_fit_set_X)
cols_to_scale = [c for c in experts_fit_set_X.columns if c not in cols_to_quantile]
desired_range = 1
scaling_factors = (
    desired_range /
    (experts_fit_set_X[cols_to_scale].max() - experts_fit_set_X[cols_to_scale].min())
)
experts_fit_set_X[cols_to_scale] *= scaling_factors

# Preprocess the deployment splits and save the transformed data
def preprocess(df):
    processed_X = df.copy()
    processed_X[cols_to_quantile] = qt.transform(processed_X[cols_to_quantile]) - 0.5  # centered on 0
    processed_X[CATEGORICAL_COLS] = oe.transform(processed_X[CATEGORICAL_COLS])
    processed_X[:] = ss.transform(processed_X)
    processed_X[cols_to_scale] *= scaling_factors
    return processed_X

if TIMESTAMP_COL is not None:
    experts_fit_set_X[TIMESTAMP_COL] = fit_set[TIMESTAMP_COL]
    experts_non_fit_set_X = preprocess(experts_non_fit_set_X)
    experts_non_fit_set_X[TIMESTAMP_COL] = non_fit_set[TIMESTAMP_COL]
    experts_fit_set_X = experts_fit_set_X.drop(columns = TIMESTAMP_COL)
    experts_non_fit_set_X = experts_non_fit_set_X.drop(columns = TIMESTAMP_COL)
else:
    experts_non_fit_set_X = preprocess(experts_non_fit_set_X)

# Synthetic Expert Generation -----------------------------------------------------------------------------------
#This function allows a user to create other groups by only defining the parameters that differ from the regular experts
def process_groups_cfg(groups_cfg, baseline_name=baseline_group):
    full_groups_cfg = dict()
    for g_name in groups_cfg:
        if g_name == baseline_name:
            full_groups_cfg[g_name] = groups_cfg[g_name]
        else:
            full_groups_cfg[g_name] = dict()
            for k in groups_cfg[baseline_name]:
                if k not in list(groups_cfg[g_name].keys()):
                    full_groups_cfg[g_name][k] = full_groups_cfg[baseline_name][k]
                elif isinstance(groups_cfg[g_name][k], dict):
                    full_groups_cfg[g_name][k] = {  # update baseline cfg
                        **groups_cfg[baseline_name][k],
                        **groups_cfg[g_name][k]
                    }
                else:
                    full_groups_cfg[g_name][k] = groups_cfg[g_name][k]

    return full_groups_cfg


ensemble_cfg = process_groups_cfg(cfg['experts']['groups'])
expert_properties_list = list()
#For each expert group generate the number of experts
for group_name, group_cfg in ensemble_cfg.items():
    try:
        np.random.seed(group_cfg['group_seed'])
    except KeyError:
        print(f"please define the 'group_seed' value for the group '{group_name}'")
        raise

    try:
        expert_seeds = np.random.randint(low = 2**32-1, size = group_cfg['n'])
    except KeyError:
        print(f"please define the 'n' value for the group '{group_name}'")

    coefs_gen = dict()

    #Verifying integrity and lack of conflicts when using w_dict
    if 'w_dict' in group_cfg:
        if not (sorted(experts_fit_set_X.columns.to_list()) == sorted(list(group_cfg['w_dict'].keys()))):
            first_set = set(sorted(experts_fit_set_X.columns.to_list()))
            sec_set = set(sorted(list(group_cfg['w_dict'].keys())))
            differences = (first_set - sec_set).union(sec_set - first_set)
            raise CustomException(f"\n\n---EXPERT GENERATION CONFIG ERROR---\n\nIf using w_dict, ensure that the two values, corresponding to the mean and stdev, are defined for every feature in the dataset.\nCurrently, the values for {differences} are not defined.")
        if ('score_mean' in group_cfg) | ('score_stdev' in group_cfg):
            raise CustomException("\n\n---EXPERT GENERATION CONFIG ERROR---\n\nIf using w_dict, define the model score weight distribution inside the dictionary\nDo not use 'score_mean' or 'score_stdev'")
        if ('protected_mean' in group_cfg) | ('protected_stdev' in group_cfg):
            raise CustomException("\\n\n---EXPERT GENERATION CONFIG ERROR---\n\nIf using w_dict, define the protected attribute's weight distribution inside the dictionary\nDo not use 'protected_mean' or 'protected_stdev'")
        if ('w_mean' in group_cfg) | ('w_stdev' in group_cfg):
            raise CustomException("\\n\n---EXPERT GENERATION CONFIG ERROR---\n\nBoth 'w_dict' and the parameters of the spike and slab distribution ('w_mean','w_stdev') are defined\nDefine EITHER 'w_dict' OR 'w_mean' and 'w_stdev'")
        coefs_gen['score'] = np.random.normal(
                    loc=0,
                    scale=0,
                    size=group_cfg['n']
            )
        coefs_gen['score'] = [None]*group_cfg['n']

        coefs_gen['protected'] = np.random.normal(
                    loc=0,
                    scale=0,
                    size=group_cfg['n']
            )
        coefs_gen['protected'] = [None]*group_cfg['n']
        group_cfg['w_stdev'] = None
        group_cfg['w_mean'] = None
    #Verifying integrity and lack of conflicts when using spike and slab
    else:
        #When using spike and slab, the score and the protected attribute can be defined sepparately, 
        #however, this can only happen if a protected attribute and/or a model score actually exist.
        #Then, we set the score values and the protected attribute values from the separate distributions, or set them to none, 
        #such that they are sampled from the spike and slab distribution within the synthetic expert object
        separately_defined_coef = []
        if MLSCORE_COL is None:
            if (f'score_mean' in group_cfg.keys()) or (f'score_stdev' in group_cfg.keys()):
                raise CustomException("\\n\n---EXPERT GENERATION CONFIG ERROR---\n\nNo Column corresponding to the model score was defined in the dataset_cfg.yaml file\nIf no model score is present, the parameters 'score_mean' and 'score_stdev' cannot be present in the cfg.yaml")
        if PROTECTED_COL is None:
            if (f'protected_mean' in group_cfg.keys()) or (f'protected_stdev' in group_cfg.keys()):
                raise CustomException("\\n\n---EXPERT GENERATION CONFIG ERROR---\n\nNo Column corresponding to the protected attribute was defined in the dataset_cfg.yaml file\nIf no model score is present, the parameters 'protected_mean' and 'protected_stdev' cannot be present in the cfg.yaml")
        for coef in ['score', 'protected']:
            if (f'{coef}_mean' in group_cfg.keys()) and (f'{coef}_stdev' in group_cfg.keys()):
                coefs_gen[coef] = np.random.normal(
                        loc=group_cfg[f'{coef}_mean'],
                        scale=group_cfg[f'{coef}_stdev'],
                        size=group_cfg['n']
                )
            else:
                coefs_gen[coef] = np.random.normal(
                        loc=0,
                        scale=0,
                        size=group_cfg['n']
                )
                coefs_gen[coef] = [None]*group_cfg['n']
        
        group_cfg['w_dict'] = None
    
    if 'theta' not in group_cfg:
            group_cfg['theta'] = 1

    if not ('alpha_mean' in group_cfg.keys()) and ('alpha_stdev' in group_cfg.keys()):
        raise CustomException("\\n\n---EXPERT GENERATION CONFIG ERROR---\n\nThe alpha parameter is necessary to generate the experts\nPlease define 'alpha_mean' and 'alpha_stdev' for the expert group.")
    else:
        if (f'alpha_mean' in group_cfg.keys()) and (f'alpha_stdev' in group_cfg.keys()):
                coefs_gen['alpha'] = np.random.normal(
                        loc=group_cfg[f'alpha_mean'],
                        scale=group_cfg[f'alpha_stdev'],
                        size=group_cfg['n']
                )
    
    coefs_spe = dict()
    coefs_spe['fnr'] = dict()
    coefs_spe['fpr'] = dict()

    #Generate the set of T_FPR, T_FNR for the group
    if ('cost' not in group_cfg) and (not(('fpr' in group_cfg) and ('fnr' in group_cfg))):
        raise CustomException("\\n\n---EXPERT GENERATION CONFIG ERROR---\n\nExpert performance metrics must be defined!\n\n-----OPTION 1: Defining Target Cost Distribution-----\nHere's an example:\n'cost':\n  'target_mean: 0.035\n  'target_stdev: 0.005\n\n-OPTION 2: Defining Target fpr and fpr Distribution-\nHere's an example:\n'fpr':\n  'target_mean: 0.50\n  'target_stdev: 0.50\n'fnr':\n  'target_mean: 0.10\n  'target_stdev: 0.10")

    if ('cost' in group_cfg) and ((('fpr' in group_cfg) or ('fnr' in group_cfg))):
        raise CustomException("\\n\n---EXPERT GENERATION CONFIG ERROR---\n\nExpert performance metrics must be defined EITHER by setting cost distribution OR by setting FPR and FNR distribution directly.\n\n-----OPTION 1: Defining Target Cost Distribution-----\nHere's an example:\n'cost':\n  'target_mean: 0.035\n  'target_stdev: 0.005\n\n-OPTION 2: Defining Target fpr and fpr Distribution-\nHere's an example:\n'fpr':\n  'target_mean: 0.50\n  'target_stdev: 0.50\n'fnr':\n  'target_mean: 0.10\n  'target_stdev: 0.10")
    
    if ('cost' not in group_cfg) and ((('fpr' in group_cfg) and ('fnr' not in group_cfg))):
        raise CustomException("\\n\n---EXPERT GENERATION CONFIG ERROR---\n\nIf Target fpr distribution is defined, Target fnr distribution must also be defined")
    
    if ('cost' not in group_cfg) and ((('fpr' not in group_cfg) and ('fnr' in group_cfg))):
        raise CustomException("\\n\n---EXPERT GENERATION CONFIG ERROR---\n\nIf Target fnr distribution is defined, Target fpr distribution must also be defined")
    
    if 'min_FNR' not in group_cfg:
        group_cfg['min_FNR'] = 0
    if 'min_FPR' not in group_cfg:
        group_cfg['min_FPR'] = 0
    if 'max_FNR' not in group_cfg:
        group_cfg['min_FNR'] = 1
    if 'max_FPR' not in group_cfg:
        group_cfg['min_FNR'] = 1

    if ('cost' in group_cfg):
        
        generated = sample_cost(group_cfg['cost']['target_mean'],
                        group_cfg['cost']['target_stdev'],
                        group_cfg['n'], 
                        prevalence, 
                        l, 
                        top_clip = group_cfg['cost']['top_clip'], 
                        bottom_clip = group_cfg['cost']['bottom_clip'], 
                        min_FNR=group_cfg['min_FNR'], 
                        max_FNR=group_cfg['max_FNR'],
                        min_FPR=group_cfg['min_FPR'], 
                        max_FPR=group_cfg['max_FPR'])
        
        coefs_spe['fnr']['target'] = generated.T[0]
        coefs_spe['fpr']['target'] = generated.T[1]
    
    if (('fpr' in group_cfg) and ('fnr' in group_cfg)):
        if not (('target_mean' in group_cfg['fpr']) and ('target_stdev' in group_cfg['fpr'])):
            raise CustomException("\\n\n---EXPERT GENERATION CONFIG ERROR---\n\nTarget fpr distribution must be defined by setting 'target_mean' and 'target_stdev' for group FPR\n\nexample:\n'fpr':\n  'target_mean: 0.50\n  'target_stdev: 0.10")
        if not (('target_mean' in group_cfg['fnr']) and ('target_stdev' in group_cfg['fnr'])):
            raise CustomException("\\n\n---EXPERT GENERATION CONFIG ERROR---\n\nTarget fnr distribution must be defined by setting 'target_mean' and 'target_stdev' for group FNR\n\nexample:\n'fnr':\n  'target_mean: 0.50\n  'target_stdev: 0.10")

        coefs_spe['fnr']['target'] = np.random.normal(
                        loc=group_cfg['fnr']['target_mean'],
                        scale=group_cfg['fnr']['target_stdev'],
                        size=group_cfg['n']
                )

        coefs_spe['fpr']['target'] = np.random.normal(
                        loc=group_cfg['fpr']['target_mean'],
                        scale=group_cfg['fpr']['target_stdev'],
                        size=group_cfg['n']
                )

    #Setting each expert's seed (for sampling of individual feature weights)
        
    for i in range(group_cfg['n']):
        expert_name = f'{group_name}#{i}'
        expert_args = dict(
            fnr_target=coefs_spe['fnr']['target'][i],
            fpr_target=coefs_spe['fpr']['target'][i],
            features_w_std = group_cfg['w_stdev'],
            features_w_mean = group_cfg['w_mean'],
            alpha = coefs_gen['alpha'][i],
            fpr_noise = 0.0,
            fnr_noise = 0.0,
            protected_w = coefs_gen['protected'][i],
            score_w = coefs_gen['score'][i],
            seed = expert_seeds[i],
            theta = group_cfg['theta'],
            features_dict = group_cfg['w_dict']
        )
        #Creating the expert objects
        expert_team[expert_name] = experts.SigmoidExpert(**expert_args)
        expert_properties_list.append({**{'expert': expert_name}, **expert_args})
        EXPERT_IDS['human_ids'].append(expert_name)


#Fitting the experts
expert_team.fit(
    X=experts_fit_set_X,
    y=fit_set[LABEL_COL],
    score_col=MLSCORE_COL,
    protected_col=PROTECTED_COL,
)

#Saving expert's properties and parameters
full_w_table = pd.DataFrame(columns = experts_fit_set_X.columns)
for expert in expert_team:
    full_w_table.loc[expert] = expert_team[expert].w

for expert in expert_team:
    full_w_table.loc[expert, 'fp_beta'] = expert_team[expert].fpr_beta
    full_w_table.loc[expert, 'fn_beta'] = expert_team[expert].fnr_beta
    full_w_table.loc[expert, 'alpha'] = expert_team[expert].alpha

os.makedirs(f'{destination_path}/', exist_ok = True)
full_w_table.to_parquet(f'{destination_path}/expert_parameters.parquet')

#Obtaining the predictions ----------------------------------------------------------------------------------

ml_train = fit_set.copy()
ml_train[CATEGORICAL_COLS] = ml_train[CATEGORICAL_COLS].astype('category')

train_expert_pred = expert_team.predict(
    index=fit_set.index,
    predict_kwargs={
        experts.SigmoidExpert: {
            'X': experts_fit_set_X,
            'y': fit_set[LABEL_COL]
        }}
)

deployment_expert_pred = expert_team.predict(
    index=non_fit_set.index,
    predict_kwargs={
        experts.SigmoidExpert: {
            'X': experts_non_fit_set_X,
            'y': non_fit_set[LABEL_COL]
        }
        
    }
)

expert_pred = pd.concat([train_expert_pred,deployment_expert_pred])

expert_pred.to_parquet(f'{destination_path}/expert_predictions.parquet')

#saving the probability of error associated with each instance
perror = pd.DataFrame()

for expert in expert_team:
    if(expert) != "model#0":
        column1 = f'p_fn_{expert}'
        column2 = f'p_fp_{expert}'
        perror[column1] = expert_team[expert].error_prob['p_of_fn']
        perror[column2] = expert_team[expert].error_prob['p_of_fp']


perror.loc[expert_pred.index].to_parquet(f'{destination_path}/prob_of_error.parquet')

with open(f'{destination_path}/expert_ids.yaml', 'w') as outfile:
    yaml.dump(EXPERT_IDS, outfile)


print('Experts generated.')