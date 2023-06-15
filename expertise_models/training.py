import os
import itertools
#hi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from joblib import Parallel, delayed
from sklearn import metrics
from aequitas.group import Group

from autodefer.models import haic
from autodefer.utils import thresholding as t, plotting

import pickle

sns.set_style('whitegrid')


cfg_path ='cfg.yaml'

with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)

RESULTS_PATH = cfg['results_path'] + cfg['exp_name'] + '/'
MODELS_PATH = cfg['models_path'] + cfg['exp_name'] + '/'

data_cfg_path = '../data/dataset_cfg.yaml'
with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

cat_dict = data_cfg['categorical_dict']


os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# import matplotlib; matplotlib.use('Agg')
width = 450
pd.set_option('display.width', width)
np.set_printoptions(linewidth=width)
pd.set_option('display.max_columns', 25)

# DATA LOADING -------------------------------------------------------------------------------------
with open(cfg['metadata'], 'r') as infile:
    metadata = yaml.safe_load(infile)

LABEL_COL = metadata['data_cols']['label']
PROTECTED_COL = metadata['data_cols']['protected']
CATEGORICAL_COLS = metadata['data_cols']['categorical']
TIMESTAMP_COL = metadata['data_cols']['timestamp']

SCORE_COL = metadata['data_cols']['score']
BATCH_COL = metadata['data_cols']['batch']
ASSIGNMENT_COL = metadata['data_cols']['assignment']
DECISION_COL = metadata['data_cols']['decision']

EXPERT_IDS = metadata['expert_ids']

TRAIN_ENVS = {
    tuple(exp_dir.split('#')): {
        'train': pd.read_parquet(cfg['train_paths']['environments'] + exp_dir + '/train.parquet'),
        'batches': pd.read_parquet(cfg['train_paths']['environments'] + exp_dir + '/batches.parquet'),
        'capacity': pd.read_parquet(cfg['train_paths']['environments'] + exp_dir + '/capacity.parquet'),
    }
    for exp_dir in os.listdir(cfg['train_paths']['environments'])
    if os.path.isdir(cfg['train_paths']['environments']+exp_dir)
}

# DEFINING FP COST ---------------------------------------------------------------------------------

with open(f'../ml_model/model/model_properties.pickle', 'rb') as infile:
    ml_model_properties = pickle.load(infile)

ml_model_threshold = ml_model_properties['threshold']
ml_model_recall = 1 - ml_model_properties['fnr']
ml_model_fpr_diff = ml_model_properties['disparity']
ml_model_fpr = ml_model_properties['fpr']

# theoretical cost
# t = fp_protected_penalty / (fp_protected_penalty + 1) <=> t.fp_protected_penalty + t = fp_protected_penalty <=> fp_protected_penalty(t-1) = -t <=> fp_protected_penalty= -t/t-1
THEORETICAL_FP_COST = -ml_model_threshold / (ml_model_threshold - 1)

# Risk Minimizing Assigners & Validation Set Construction ------------------------------------------
VAL_ENVS = dict()
VAL_X = None
RMAs = dict()
for env_id in TRAIN_ENVS:
    batch_id, capacity_id = env_id
    models_dir = f'{MODELS_PATH}{batch_id}_{capacity_id}/'
    os.makedirs(models_dir, exist_ok=True)

    train_with_val = TRAIN_ENVS[env_id]['train']
    train_with_val = train_with_val.copy().drop(columns=BATCH_COL)  # not needed
    is_val = (train_with_val[TIMESTAMP_COL] == 6)
    train_with_val = train_with_val.drop(columns=TIMESTAMP_COL)
    train = train_with_val[~is_val].copy()
    val = train_with_val[is_val].copy()
    #Here it trains the human models and assigners for each of the settings
    RMAs[env_id] = haic.assigners.RiskMinimizingAssigner(
        expert_ids=EXPERT_IDS,
        outputs_dir=f'{models_dir}human_expertise_model/',
    )

    RMAs[env_id].fit(
        train=train,
        val=val,
        categorical_cols=CATEGORICAL_COLS, score_col=SCORE_COL,
        decision_col=DECISION_COL, ground_truth_col=LABEL_COL, assignment_col=ASSIGNMENT_COL,
        hyperparam_space=cfg['human_expertise_model']['hyperparam_space'],
        n_trials=cfg['human_expertise_model']['n_trials'],
        random_seed=cfg['human_expertise_model']['random_seed'], 
        CAT_DICT = cat_dict
    )

    VAL_ENVS[env_id] = dict()
    if VAL_X is None:  # does not change w/ env
        VAL_X_COMPLETE = val.copy()
        VAL_X = VAL_X_COMPLETE.copy().drop(columns=[ASSIGNMENT_COL, DECISION_COL, LABEL_COL])
    VAL_ENVS[env_id]['batches'] = (
        TRAIN_ENVS[env_id]['batches']
        .loc[val.index, ]
        .copy()
    )
    VAL_ENVS[env_id]['capacity'] = (
        TRAIN_ENVS[env_id]['capacity']
        .loc[VAL_ENVS[env_id]['batches']['batch'].unique(), ]
        .copy()
    )

# Evaluate Human Expertise Models ------------------------------------------------------------------
def get_outcome(label, pred):
    if pred == 1:
        if label == 1:
            o = 'tp'
        elif label == 0:
            o = 'fp'
    elif pred == 0:
        if label == 1:
            o = 'fn'
        elif label == 0:
            o = 'tn'
    return o

OUTCOME_COL = 'error'
expert_val_X = VAL_X_COMPLETE.copy()
expert_val_X = expert_val_X[expert_val_X[ASSIGNMENT_COL] != EXPERT_IDS['model_ids'][0]]
expert_val_X[OUTCOME_COL] = expert_val_X.apply(
    lambda x: get_outcome(label=x[LABEL_COL], pred=x[DECISION_COL]),
    axis=1,
)
expert_val_X = expert_val_X.drop(columns=[DECISION_COL, LABEL_COL])

# EVALUATION FUNCTIONS -----------------------------------------------------------------------------
def make_id_str(tpl):
    printables = list()
    for i in tpl:
        if i == '':
            continue
        elif isinstance(i, (bool, int, float)):
            printables.append(str(i))
        else:
            printables.append(i)

    return '_'.join(printables)


def product_dict(**kwargs):  # aux
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def make_params_combos(params_cfg):
    params_list = list()
    if not isinstance(params_cfg, list):
        params_cfg = [params_cfg]

    for cartesian_product_set in params_cfg:
        for k, v in cartesian_product_set.items():
            if isinstance(v, str):
                cartesian_product_set[k] = [v]
        for p in product_dict(**cartesian_product_set):
            p_params = {**BASE_CFG, **p}
            if p_params['fp_cost'] == 'theoretical':
                p_params['fp_cost'] = THEORETICAL_FP_COST
            if not (
                p_params['calibration'] and  # useless to calibrate in these cases
                (p_params['confidence_deferral'] or p_params['solver'] == 'random')
            ):
                params_list.append(p_params)

    return params_list

def make_assignments(X, envs, rma, exp_params):
    env_id = (exp_params['batch'], exp_params['capacity'])
    assigner_params = {k: v for k, v in exp_params.items() if k not in ['batch', 'capacity']}
    params_to_record = {k: exp_params[k] for k in FIELDS}
    exp_id = tuple([v for k, v in params_to_record.items()])
    print(exp_id)
    a = rma.assign(
        X=X, score_col=SCORE_COL,
        batches=envs[env_id]['batches'],
        capacity=envs[env_id]['capacity'].T.to_dict(),
        ml_model_threshold=ml_model_threshold,
        protected_col=(X[PROTECTED_COL] >= 50).map({True: 'Older', False: 'Younger'}),
        protected_group='Older',
        assignments_relative_path=make_id_str(exp_id),
        **assigner_params
    )

    return exp_id, assigner_params, a

def predicted_evaluation(X, assignments, rma, fp_cost):
    X = X.copy().assign(**{ASSIGNMENT_COL: assignments})
    X[ASSIGNMENT_COL] = X[ASSIGNMENT_COL].astype('category')
    X['index'] = X.index

    print('hi')

    pred_out_proba = rma.predict_outcome_probabilities(
        X=X, score_col=SCORE_COL,
        ml_model_threshold=ml_model_threshold,
        calibration=True
    )

    print('hi2')

    pred_out_proba = rma.predict_outcome_probabilities(
        X=X, score_col=SCORE_COL,
        ml_model_threshold=ml_model_threshold,
        calibration=True
    )
    loss = fp_cost * pred_out_proba['fp'].sum() + pred_out_proba['fn'].sum()
    tpr = pred_out_proba['tp'].sum() / (pred_out_proba['tp'].sum() + pred_out_proba['fn'].sum())
    fpr = pred_out_proba['fp'].sum() / (pred_out_proba['tn'].sum() + pred_out_proba['fp'].sum())

    protected_col = (X[PROTECTED_COL] >= 50).map({True: 'Older', False: 'Younger'})
    is_protected_bool = (protected_col == 'Older')
    fpr_disparity = (
        (pred_out_proba[~is_protected_bool]['fp'].sum()
           / (pred_out_proba[~is_protected_bool]['tn'].sum()
              + pred_out_proba[~is_protected_bool]['fp'].sum()))
        / (pred_out_proba[is_protected_bool]['fp'].sum()
           / (pred_out_proba[is_protected_bool]['tn'].sum()
              + pred_out_proba[is_protected_bool]['fp'].sum()))
    )
    return loss, tpr, fpr, fpr_disparity

def make_assignments_and_predict_evaluate(X, envs, rma, exp_params):
    exp_id, assigner_params, a = make_assignments(X=X, envs=envs, rma=rma, exp_params=exp_params)
    pred_loss, pred_tpr, pred_fpr, pred_fpr_disparity = predicted_evaluation(
        X=X, assignments=a, rma=rma, fp_cost=assigner_params['fp_cost'],
    )
    return exp_id, pred_loss, pred_tpr, pred_fpr, pred_fpr_disparity


ENV_FIELDS = ['batch', 'capacity']
ASSIGNER_FIELDS = [
    'confidence_deferral', 'solver', 'calibration', 'fp_cost', 'fp_protected_penalty',
    'dynamic', 'target_fpr_disparity', 'fpr_learning_rate', 'fpr_disparity_learning_rate'
]
FIELDS = ENV_FIELDS + ASSIGNER_FIELDS
print(tuple(FIELDS))

BASE_CFG = cfg['base_cfg']

#Leo: So i guess if the experiments are made under the various environments, what Diogo does
#Leo: is to test the results in the validation dataset, where the assignments and predictions are done by us
#Leo: So what Pedro wants is to check how the validation loss changes with the capacity imposed to the 
#Leo: assigner ONLY on the validation set. (either for batch or online? both?)
#Leo: So which of the training regiments do we choose? Regular? Scarce?
#Leo: Anyway they all perform similarly (on average) when predicting the expert's probability of error, so it doesn't matter (?)
#Leo: Maybe it's terrible at modeling SOME of the experts, so it can make the whole system worse.

# EXPERIMENTS --------------------------------------------------------------------------------------
print("----Experiments start----\n")
val_results_dict = dict()
if cfg['n_jobs'] > 1:
    Parallel(n_jobs=cfg['n_jobs'])(
        delayed(make_assignments)(
            X=VAL_X,
            envs=VAL_ENVS,
            rma=RMAs[(exp_params['batch'], exp_params['capacity'])],
            exp_params=exp_params
        )
        for exp_params in make_params_combos(cfg['experiments'])
    )




for exp_params in make_params_combos(cfg['experiments']):
    exp_id, pred_loss, pred_tpr, pred_fpr, pred_fpr_disparity = (
        make_assignments_and_predict_evaluate( #Line: 359
            X=VAL_X,
            envs=VAL_ENVS,
            rma=RMAs[(exp_params['batch'], exp_params['capacity'])],
            exp_params=exp_params)
    )#Tipo aqui o diogo usa os modelos de cada um dos training environments. não muda só o deployment. 
    val_results_dict[exp_id] = dict(
        pred_loss=pred_loss, pred_tpr=pred_tpr, pred_fpr=pred_fpr,
        pred_fpr_disparity=pred_fpr_disparity
    )

val_results = pd.DataFrame(val_results_dict).T.reset_index(drop=False)
val_results.columns = FIELDS + ['pred_loss', 'pred_tpr', 'pred_fpr', 'pred_fpr_disparity']
val_results.to_parquet('val_results.parquet')

val_results = val_results.drop(
    columns=['dynamic', 'target_fpr_disparity', 'fpr_learning_rate', 'fpr_disparity_learning_rate']
)
# RENAME FOR PLOTS
col_renamings = {
    'batch': 'Batch',
    'capacity': 'Capacity',
    'confidence_deferral': 'Confidence Deferral',
    'calibration': 'Calibration',
    'solver': 'Solver',
    'fp_cost': 'lambda',
    'fp_protected_penalty': 'alpha',
    'pred_loss': 'Loss',
    'pred_fpr': 'Predicted FPR',
    'pred_tpr': 'Predicted TPR',
    'pred_fpr_disparity': 'Predicted FPR Parity'
}

architecture_results = val_results[
    (val_results['fp_cost'] == THEORETICAL_FP_COST) &
    (val_results['fp_protected_penalty'] == 0)
]
architecture_results = architecture_results[
    ((architecture_results['confidence_deferral']) & (architecture_results['solver'] == 'random'))
    | ((architecture_results['confidence_deferral'] == False) & (architecture_results['solver'] != 'random'))
]
(
    architecture_results
    .groupby(['confidence_deferral', 'solver', 'calibration'])
    .mean()
    .sort_values(by='pred_loss')
    .reset_index()
)

architecture_results['Method'] = (
    architecture_results['confidence_deferral'].map({
        True: 'Model-Confidence Deferral',
        False: 'Learning to Assign'
    })
)
plot_data = architecture_results[architecture_results['solver'] != 'random'].copy()
plot_data = plot_data.rename(columns={'calibration': 'Calibration'})
plot_data = (
    plot_data
    .replace('individual', 'Greedy \n (instance-based)')
    .replace('scheduler', 'Linear Programming \n (batch-based)')
)
sns.stripplot(
    data=plot_data, x='solver', y='pred_loss',
    hue='Calibration'
)
plt.ylim(bottom=0)
plt.xlabel('')
plt.ylabel('Predicted Loss')
plt.show()
"""
sns.scatterplot(
    data=architecture_results, x='Method', y='pred_loss',
    hue='calibration', style='solver',
    alpha=0
)
handles, labels = plt.gca().get_legend_handles_labels()
greedy = architecture_results[architecture_results['solver'] == 'greedy']
m = sns.stripplot(
    data=greedy, x='solver', y='pred_loss', hue='calibration',
    marker='o', edgecolor='grey', jitter=1,
)

scheduler = architecture_results[architecture_results['solver'] == 'scheduler']
n = sns.stripplot(
    data=scheduler, x='solver', y='pred_loss', hue='calibration',
    marker='X', edgecolor='grey', jitter=1,
)
plt.legend(handles, labels)
plt.show()
"""

fp_cost_results = val_results[
    (val_results['confidence_deferral'] == False) &
    (val_results['solver'] == 'scheduler') &
    (val_results['calibration'] == True) &
    (val_results['fp_protected_penalty'] == 0)
]

(
    fp_cost_results
    .pivot(index='fp_cost', columns=['batch', 'capacity'], values='pred_fpr')
    .T.reset_index()
)
sns.lineplot(
    data=fp_cost_results[fp_cost_results['fp_cost'].isin([THEORETICAL_FP_COST, 0.05, 1, 2])],
    x='fp_cost', y='pred_fpr', markers=True,
    hue='capacity', style='batch',
    palette='colorblind'
)
plt.show()

plot_data = fp_cost_results[fp_cost_results['fp_cost'] < 1]
plot_data = plot_data.rename(columns={'capacity': 'Capacity', 'batch': 'Batch'})
plot_data = (
    plot_data
    .replace('regular', 'Regular')
    .replace('inconstant', 'Inconstant')
    .replace('model_dominant', 'Human-Scarce')
    .replace('irregular', 'Disparate')
    .replace('small', 'Small')
    .replace('large', 'Large')
)
sns.lineplot(
    data=plot_data,
    x='fp_cost', y='pred_fpr', markers=True,
    hue='Capacity', style='Batch',
    palette='colorblind',
)
plt.xlabel(r'$\lambda$')
plt.ylabel('Predicted FPR')
plt.axhline(cfg['fpr'], linestyle='dashed', color='grey')
plt.show()

fairness_results = val_results[
    (val_results['confidence_deferral'] == False)
    & (val_results['solver'] == 'scheduler')
    & (val_results['calibration'] == True)
]
sns.scatterplot(
    data=fairness_results,
    x='pred_fpr',
    y='pred_tpr',
    hue='fp_protected_penalty'
)
plt.show()

# fairness_results['violation'] = (fairness_results['pred_fpr'] - cfg['fpr']).abs()
fairness_results_below_fpr = fairness_results[fairness_results['pred_fpr'] <= cfg['fpr']]
fairness_results_below_fpr[
    (fairness_results_below_fpr['batch'] == 'large')
    & (fairness_results_below_fpr['capacity'] == 'inconstant')
].sort_values(by=['fp_protected_penalty', 'pred_fpr'])
fairness_results_at_fpr = (
    fairness_results_below_fpr
    # .sort_values(by='violation', ascending=True)
    .sort_values(by='pred_fpr', ascending=False)
    .groupby(['batch', 'capacity', 'fp_protected_penalty'])
    .head(1)
    .sort_values(by=['batch', 'capacity', 'fp_protected_penalty'])
)
plot_data = fairness_results_at_fpr
plot_data = plot_data.rename(columns={'capacity': 'Capacity', 'batch': 'Batch'})
plot_data = (
    plot_data
    .replace('regular', 'Regular')
    .replace('inconstant', 'Inconstant')
    .replace('model_dominant', 'Human-Scarce')
    .replace('irregular', 'Disparate')
    .replace('small', 'Small')
    .replace('large', 'Large')
)
sns.lineplot(
    data=plot_data,
    x='pred_tpr', y='pred_fpr_disparity', markers=True,
    hue='Capacity', style='Batch',
    palette='colorblind',
    sort=False,
)
plt.xlabel('Predicted TPR')
plt.ylabel('Predicted FPR Parity')
plt.xlim(0.5, 0.7)
plt.ylim(0, 1)
plt.show()