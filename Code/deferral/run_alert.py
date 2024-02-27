import numpy as np
import pandas as pd
import pickle
import yaml
import os
from ortools.sat.python import cp_model
import random
from joblib import Parallel, delayed

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

def full_auto_func(capacities, batches, testset, env, model):
    print(f'solving {env}: fullauto')
    if os.path.isdir(f'../../FiFAR/deferral/results/{model}/'  + env):
        return

    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    for i in np.arange(1,batches['batch'].max()+1):
        cases = testset.loc[batches.loc[batches['batch'] == i]['case_id'],:]
        for ix in cases.index:
            assignments.loc[ix] = 'auto-reject'
            results.loc[ix] = 1
        
            
    if not os.path.isdir(f'../../FiFAR/deferral/results/{model}/'  + env):
        os.makedirs(f'../../FiFAR/deferral/results/{model}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'../../FiFAR/deferral/results/{model}/' + env +'/assignments.parquet')
    results.to_parquet(f'../../FiFAR/deferral/results/{model}/' + env +'/results.parquet')

    return assignments, results

def full_model_func(capacities, batches, testset, expert_preds, env, model):
    print(f'solving {env}: fullauto')
    if os.path.isdir(f'../../FiFAR/deferral/results/{model}/'  + env):
        return

    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    for i in np.arange(1,batches['batch'].max()+1):
        cases = testset.loc[batches.loc[batches['batch'] == i]['case_id'],:]
        for ix in cases.index:
            assignments.loc[ix] = 'classifier_h'
            results.loc[ix] = expert_preds.loc[ix,'classifier_h']
        
            
    if not os.path.isdir(f'../../FiFAR/deferral/results/{model}/'  + env):
        os.makedirs(f'../../FiFAR/deferral/results/{model}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'../../FiFAR/deferral/results/{model}/' + env +'/assignments.parquet')
    results.to_parquet(f'../../FiFAR/deferral/results/{model}/' + env +'/results.parquet')

    return assignments, results

def rand_deferral_func(capacities, batches, testset, expert_preds, env, model, seed):
    print(f'solving {env}: rand')
    if os.path.isdir(f'../../FiFAR/deferral/results/{model}/{seed}/'  + env):
        return
    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    random.seed(seed)
    for i in np.arange(1,batches['batch'].max()+1):
        c = capacities.loc[capacities['batch_id'] == i,:].iloc[0]
        c.loc['classifier_h'] = c['batch_size'] - c[2:].sum()
        c = c.loc[c != 0]
        cases = testset.loc[batches.loc[batches['batch'] == i]['case_id'],:]
        expert_preds = expert_preds.loc[:,c.index.drop(['batch_id','batch_size'])]
        experts = expert_preds.columns.to_list()
        for ix in cases.index:
            done = 0
            while (done != 1):
                choice  = random.choice(experts)
                if c[choice]>0:
                    c[choice] -= 1
                    assignments.loc[ix] = choice
                    results.loc[ix] = expert_preds.loc[ix, choice]
                    done = 1
                else:
                    experts.remove(choice)
                if len(choice) == 0:
                    done = 1
            
    if not os.path.isdir(f'../../FiFAR/deferral/results/{model}/{seed}/' + env):
        os.makedirs(f'../../FiFAR/deferral/results/{model}/{seed}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'../../FiFAR/deferral/results/{model}/{seed}/' + env +'/assignments.parquet')
    results.to_parquet(f'../../FiFAR/deferral/results/{model}/{seed}/' + env +'/results.parquet')
    
    return assignments, results

def ova_deferral_func(capacities, batches, testset, expert_preds, model_preds, env, model, seed, team):
    team1 = seed.split('#')[1]
    if team1 != team:
        return

    print(f'Calculating OvA for {env}')
    if os.path.isdir(f'../../FiFAR/deferral/results/{model}/{seed}/'  + env):
        return
    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    for i in np.arange(1,batches['batch'].max()+1):
        c = capacities.loc[capacities['batch_id'] == i,:].iloc[0]
        c.loc['classifier_h'] = c['batch_size'] - c[2:].sum()
        c = c.loc[c != 0]
        cases = testset.loc[batches.loc[batches['batch'] == i]['case_id'],:]
        preds = model_preds.loc[cases.index]
        preds = preds.loc[:,c.index.drop(['batch_id','batch_size'])]
        for ix, row in preds.iterrows():
            sorted = row.sort_values(ascending = False)
            for choice in sorted.index:
                if c[choice]>0:
                    c[choice] -= 1
                    assignments.loc[ix] = choice
                    results.loc[ix] = expert_preds.loc[ix, choice]
                    break

    if not os.path.isdir(f'../../FiFAR/deferral/results/{model}/{seed}/' + env):
        os.makedirs(f'../../FiFAR/deferral/results/{model}/{seed}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'../../FiFAR/deferral/results/{model}/{seed}/' + env +'/assignments.parquet')
    results.to_parquet(f'../../FiFAR/deferral/results/{model}/{seed}/' + env +'/results.parquet')
    return assignments, results

def deccaf_deferral_func(capacities, batches, testset, expert_preds, model_preds, env, model, seed, team):
    team1 = seed.split('#')[1]
    if team1 != team:
        return
    
    if os.path.isdir(f'../../FiFAR/deferral/results/{model}/{seed}/'  + env):
        return

    model_name = model
    print(f'solving linear {model} for {env}')
    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    for b in np.arange(1,batches['batch'].max()+1):
        c = capacities.loc[capacities['batch_id'] == b,:].iloc[0]
        c.loc['classifier_h'] = c['batch_size'] - c[2:].sum()
        c = c.loc[c != 0]
        cases = testset.loc[batches.loc[batches['batch'] == b]['case_id'],:] 
        preds = model_preds.loc[cases.index]
        preds = preds.loc[:,c.index.drop(['batch_id','batch_size'])]
        cost_matrix_df = preds.T
        for d in c.index:
            if c.loc[d] == 0:
                cost_matrix_df = cost_matrix_df.drop(index=d)
                

        cost_matrix = cost_matrix_df.values
        num_workers, num_tasks = cost_matrix.shape
        workers = list(cost_matrix_df.index)

        model = cp_model.CpModel()
        x = []
        for i in range(num_workers):
            t = []
            for j in range(num_tasks):
                t.append(model.NewBoolVar(f'x[{i},{j}]'))
            x.append(t)

        # capacity constraints
        for i in range(num_workers):
            model.Add(sum([x[i][j] for j in range(num_tasks)]) == c[workers[i]])

        # Each task is assigned to exactly one worker.
        for j in range(num_tasks):
            model.AddExactlyOne(x[i][j] for i in range(num_workers))

        objective_terms = []
        for i in range(num_workers):
            for j in range(num_tasks):
                objective_terms.append(cost_matrix[i, j] * x[i][j])
        model.Maximize(sum(objective_terms))
        #This sum(objective_terms) is the loss of the batch.
        solver = cp_model.CpSolver()
        # solver.parameters.log_search_progress = True
        solver.parameters.max_time_in_seconds = 60.0
        status = solver.Solve(model)

        if not status == cp_model.OPTIMAL and not status == cp_model.FEASIBLE:
            print('Solution not found!')
            return None

        print('Batch solved')
        
        for j in range(num_tasks):
            ix = cost_matrix_df.columns.to_list()[j]
            for i in range(num_workers):
                if solver.BooleanValue(x[i][j]):
                    assignments.loc[ix] = workers[i]
                    results.loc[ix] = expert_preds.loc[ix, workers[i]]

    
    
    if not os.path.isdir(f'../../FiFAR/deferral/results/{model_name}/{seed}/' + env):
        os.makedirs(f'../../FiFAR/deferral/results/{model_name}/{seed}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'../../FiFAR/deferral/results/{model_name}/{seed}/' + env +'/assignments.parquet')
    results.to_parquet(f'../../FiFAR/deferral/results/{model_name}/{seed}/' + env +'/results.parquet')

    return assignments, results


data_cfg_path = '../alert_data/dataset_cfg.yaml'
with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

cat_dict = data_cfg['categorical_dict']

alerts = pd.read_parquet(f'../../FiFAR/alert_data/processed_data/alerts.parquet')


test = alerts.loc[alerts['month'] == 7]
exp_pred = pd.read_parquet(f'../../FiFAR/synthetic_experts/expert_predictions.parquet').loc[test.index]


with open(f"../../FiFAR/classifier_h/selected_model/best_model.pickle", 'rb') as fp:
    classifier_h = pickle.load(fp)
with open(f"../../FiFAR/classifier_h/selected_model/model_properties.yaml", 'r') as fp:
    classifier_h_properties = yaml.safe_load(fp)

def sig(x):
    return 1/(1+np.exp(-x))

def output(data, model, init_score):
    return sig(model.predict(data,raw_score=True) + init_score)

X_test = test.drop(columns = ['fraud_bool','model_score','month']) 

h_preds = output(X_test, classifier_h, classifier_h_properties['init_score'])

exp_pred['classifier_h'] = h_preds

exp_pred = (exp_pred>=0.5).astype(int)
CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

with open(f'../../FiFAR/deferral/l2d_predictions/ova.pkl', 'rb') as fp:
        ova_model_preds = pickle.load(fp)

with open(f'../../FiFAR/deferral/l2d_predictions/deccaf.pkl', 'rb') as fp:
        deccaf_model_preds = pickle.load(fp)

a = dict()
for direc in os.listdir(f'../../FiFAR/testbed/test'):
    if os.path.isfile(f'../../FiFAR/testbed/test/' + direc):
        continue
    a[direc] = dict()
    a[direc]['bat'] = pd.read_csv(f'../../FiFAR/testbed/test/' + direc + '/batches.csv')
    a[direc]['cap'] = pd.read_csv(f'../../FiFAR/testbed/test/' + direc + '/capacity.csv')
    a[direc]['team'] = direc.split('#')[1].split('-')[0]

for seed in deccaf_model_preds:
    Parallel(n_jobs=5)(
            delayed(deccaf_deferral_func)(
                a[env]['cap'],
                a[env]['bat'],
                test,
                exp_pred,
                deccaf_model_preds[seed],
                env,
                f'DeCCaF',
                seed,
                a[env]['team']
            )
            for env in a 
        )
    
for seed in ova_model_preds:
    Parallel(n_jobs=5)(
            delayed(ova_deferral_func)(
                a[env]['cap'],
                a[env]['bat'],
                test,
                exp_pred,
                ova_model_preds[seed],
                env,
                f'OvA',
                seed,
                a[env]['team']
            )
            for env in a 
        )


Parallel(n_jobs=5)(
            delayed(full_auto_func)(
                a[env]['cap'],
                a[env]['bat'],
                test,
                env,
                f'Full_Rej'
            )
            for env in a 
        )

Parallel(n_jobs=5)(
            delayed(full_model_func)(
                a[env]['cap'],
                a[env]['bat'],
                test,
                exp_pred,
                env,
                f'Only_Classifier'
            )
            for env in a 
        )

for seed in [1,2,3,4,5]:
    Parallel(n_jobs=5)(
                delayed(rand_deferral_func)(
                    a[env]['cap'],
                    a[env]['bat'],
                    test,
                    exp_pred,
                    env,
                    f'Random',
                    seed
                )
                for env in a 
            )

