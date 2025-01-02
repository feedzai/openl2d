# %%
import os
import random
import numpy as np
import pandas as pd
import yaml

class CustomException(Exception):
    pass

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

with open('cfg.yaml', 'r') as infile:
    cfg = yaml.safe_load(infile)

with open(cfg['data_cfg_path'], 'r') as infile:
    data_cfg = yaml.safe_load(infile)


np.random.seed(cfg['random_seed'])
random.seed(cfg['random_seed'])


# DATA LOADING -------------------------------------------------------------------------------------
data = pd.read_parquet(cfg['dataset_path'])

if 'categorical' in data_cfg['data_cols']:
    CATEGORICAL_COLS = data_cfg['data_cols']['categorical']
    data[CATEGORICAL_COLS] = data[CATEGORICAL_COLS].astype('category')
    if 'categorical_dict' not in data_cfg:
        raise CustomException("Please define the categorical feature dictionary 'categorical_dict' in the dataset's configuration file.")
    cat_dict = data_cfg['categorical_dict']
    data = cat_checker(data, CATEGORICAL_COLS, cat_dict)

with open(f'{cfg["expert_folder_path"]}/expert_ids.yaml', 'r') as infile:
    EXPERT_IDS = yaml.safe_load(infile)
    EXPERT_CATS = EXPERT_IDS['human_ids']

try:
    training_set = cfg['training_set']
except KeyError:
    print("Please define the training_set in the file 'cfg.yaml'")
    raise

if 'timestamp' in data_cfg['data_cols']:
    TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
    train = data.loc[((data[TIMESTAMP_COL] >= training_set[0]) & (data[TIMESTAMP_COL] < training_set[1]))]
else:
    train = data.loc[training_set[0]:training_set[1]-1,:]
    TIMESTAMP_COL = None

train = cat_checker(train, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

# EXPERTS ------------------------------------------------------------------------------------------
# produced in experts/experts_generation.py
experts_pred = pd.read_parquet(f'{cfg["expert_folder_path"]}/expert_predictions.parquet')
train_expert_pred = experts_pred.loc[train.index, ]
# BATCH & CAPACITY ---------------------------------------------------------------------------------
#--------------------------------------------
for b_cfg in cfg['environments_train']['batch']:
    properties = cfg['environments_train']['batch'][b_cfg]
    if 'size' not in properties:
        raise CustomException(f"\n\n-----------TRAINING BATCH PROPERTIES CONFIG ERROR------------\n\n'size' parameter must be defined for all batch settings - this was not done in setting '{b_cfg}'")
    if 'seed' not in properties:
        raise CustomException(f"\n\n-----------TRAINING BATCH PROPERTIES CONFIG ERROR------------\n\n'seed' parameter must be defined for all batch settings - this was not done in setting '{b_cfg}'")


for c_cfg in cfg['environments_train']['capacity']:
    properties = cfg['environments_train']['capacity'][c_cfg]
    if 'deferral_rate' not in properties:
        raise CustomException(f"\n\n-----------TRAINING CAPACITY PROPERTIES CONFIG ERROR------------\n\n'deferral_rate' parameter must be defined for all capacity settings - this was not done in setting '{c_cfg}'")
    if 'distribution' not in properties:
        raise CustomException(f"\n\n-----------TRAINING CAPACITY PROPERTIES CONFIG ERROR------------\n\n'distribution' parameter must be defined for all capacity settings - this was not done in setting '{c_cfg}'")
    else:
        if properties['distribution'] not in ['homogeneous','variable']:
            raise CustomException(f"\n\n-----------TRAINING CAPACITY PROPERTIES CONFIG ERROR------------\n\n'distribution' parameter must be either 'homogeneous' or 'variable' - this was not done in setting '{c_cfg}'")
        if properties['distribution'] == 'variable':
            if ('distribution_stdev' not in properties) or ('distribution_seed' not in properties) or ('variable_capacity_per_batch' not in properties):
                raise CustomException(f"\n\n-----------TRAINING CAPACITY PROPERTIES CONFIG ERROR------------\n\n If 'distribution' is set to 'variable', 'distribution_stdev', 'distribution_seed' and 'variable_capacity_per_batch' must be defined - this was not done in setting '{c_cfg}'")
    if 'n_experts' in properties:
        if (properties['n_experts'] < 1) or (properties['n_experts'] > len(experts_pred.columns)):
            raise CustomException(f"\n\n-----------TRAINING CAPACITY PROPERTIES CONFIG ERROR------------\n\n If 'n_experts' is set, it must be >=1 and <= Total number of experts - check setting '{c_cfg}'")
        if ('n_experts_seed' not in properties) or ('variable_experts_per_batch' not in properties):
            raise CustomException(f"\n\n-----------TRAINING CAPACITY PROPERTIES CONFIG ERROR------------\n\n If 'n_experts' is set, you must also set the value for 'n_experts_seed' and 'variable_experts_per_batch' - check setting '{c_cfg}'")



def generate_batches(df, batch_properties: dict, timestamp) -> pd.DataFrame:
    """
    Generates a pandas dataframe indicating the (serial) number of the batch each instance belongs to.
    Batches do not crossover from one month to the other.
    :param batch_properties: dictionary containing size key-value pair (see cfg.yaml).
    :param months: pandas series indicating the month of each instance.
    """
    batches_timestamp_list = list()
    last_batch_ix = 0
    for m in timestamp.unique():
        df_m = df[timestamp == m]
        df_m = df_m.sample(frac = 1, random_state = batch_properties['seed'])
        
        m_batches = pd.DataFrame(
            [int(i / batch_properties['size']) + last_batch_ix + 1 for i in range(len(df_m))],
            index=df_m.index,
            columns=['batch'],
        )
        batches_timestamp_list.append(m_batches)
        last_batch_ix = int(m_batches.max())

    batches = pd.concat(batches_timestamp_list)

    return batches

def generate_capacity_single_batch(batch_size: int, properties: dict, human_ids: list, batch_id: int) -> dict:
    """
    Generates dictionary indicating the capacity of each decision-maker (from model_id and human_ids).
    This capacity pertains to a single batch.
    :param properties: dictionary indicating capacity constraints (see cfg.yaml)
    :param model_id: identification of the model to be used in the output dictionary.
    :param human_ids: identification of the humans to be used in the output dictionary.
    """
    capacity_dict = dict()

    if properties['distribution'] == 'homogeneous':
        humans_capacity_value = int(
            int(batch_size*properties['deferral_rate']) /
            len(human_ids)
        )
        unc_human_capacities = np.full(shape=(len(human_ids),), fill_value=humans_capacity_value)

    elif properties['distribution'] == 'variable':  # capacity follows a random Gaussian

        if properties['variable_capacity_per_batch']:
            random.seed(properties['distribution_seed'] + batch_id)
        else:
            random.seed(properties['distribution_seed'])

        mean_individual_capacity = (batch_size ) / len(human_ids)
        unc_human_capacities = np.random.normal(
            loc=mean_individual_capacity,
            scale=properties['distribution_stdev'] * mean_individual_capacity,
            size=(len(human_ids),),
        )
        unc_human_capacities += (
            (batch_size - sum(unc_human_capacities))
            / len(human_ids)
        )

    available_humans_ix = list(range(len(human_ids)))
    if 'n_experts' in properties:  # some experts are randomly unavailable

        if properties['variable_experts_per_batch']:
            random.seed(properties['n_experts_seed'] + batch_id)
        else:
            random.seed(properties['n_experts_seed'])

        absent_humans_ix = random.sample(  # without replacement
            available_humans_ix,
            k=len(available_humans_ix) - int(properties['n_experts']),
        )
        unc_human_capacities[absent_humans_ix] = 0

        unassigned = (int(batch_size*properties['deferral_rate'])   - sum(unc_human_capacities))
        available_humans_ix = [ix for ix in available_humans_ix if ix not in absent_humans_ix]
        unc_human_capacities = unc_human_capacities.astype(float)
        unc_human_capacities[available_humans_ix] *= (1 + unassigned / sum(unc_human_capacities))

    # convert to integer and adjust for rounding errors
    human_capacities = np.floor(unc_human_capacities).astype(int)
    unassigned = int(int(batch_size*properties['deferral_rate'])  - sum(human_capacities))
    assert unassigned < len(human_ids)
    random.seed(42)
    to_add_to = random.sample(available_humans_ix, k=unassigned)
    human_capacities[to_add_to] += 1

    capacity_dict.update(**{
        human_ids[ix]: int(human_capacities[ix])
        for ix in range(len(human_ids))
    })

    assert sum(capacity_dict.values()) == int(batch_size*properties['deferral_rate']) 

    return capacity_dict

def generate_capacity(batches: pd.Series, capacity_properties: dict) -> pd.DataFrame:
    """
    Generates pandas dataframe matching batch_ids to capacity constraints for that batch.
    :param batches: pandas dataframe output by generate_batches()
    :param capacity_properties: dictionary output by generate_capacity_single_batch()
    """
    capacity_df = pd.DataFrame.from_dict(
        {
            int(b_ix): generate_capacity_single_batch(
                batch_size=int((batches == b_ix).sum()),
                properties=capacity_properties,
                human_ids=EXPERT_IDS['human_ids'],
                batch_id = b_ix
            )
            for b_ix in batches.iloc[:, 0].unique()
        },
        orient='index'
    )
    return capacity_df

def generate_environments(df, batch_cfg: dict, capacity_cfg: dict, output_dir=None) -> dict:
    """
    Generates a dictionary matching environment keys to batch and capacity dataframes.
    :param batch_cfg: dictionary with the batch configurations (see cfg.yaml).
    :param capacity_cfg: dictionary with the capacity configurations (see cfg.yaml).
    :param output_dir: directory to save to.
    """
    environments = dict()
    for batch_scheme, batch_properties in batch_cfg.items():
        for capacity_scheme, capacity_properties in capacity_cfg.items():
            print(f'Generating environments for the combination {batch_scheme},{capacity_scheme}')
            if ('timestamp' in data_cfg['data_cols']) and cfg['timestamp_constraint']:
                batches_df = generate_batches(
                    df=df,
                    batch_properties=batch_properties,
                    timestamp=df[data_cfg['data_cols']['timestamp']]
                )
            else:
                batches_df = generate_batches(
                        df=df,
                        batch_properties=batch_properties,
                        timestamp=pd.Series(index = df.index, data = np.zeros(len(df)))
                    )
            capacity_df = generate_capacity(
                batches=batches_df, capacity_properties=capacity_properties)
            if output_dir is not None:
                env_path = f'{output_dir}{batch_scheme}#{capacity_scheme}/'
                os.makedirs(env_path, exist_ok=True)
                batches_df.to_csv(env_path+'batches.csv')
                capacity_df.to_csv(env_path+'capacity.csv')
            environments[(batch_scheme, capacity_scheme)] = (batches_df, capacity_df)

    return environments

def generate_predictions(X, expert_pred, batches, capacity, output_dir=None):
    """
    Randomly assigns instances to decision-makers. Queries said decision-makers for decisions.
    Returns X dataframe merged with assignments and decisions.
    :param X: full dataset, including features.
    :param expert_pred: full matrix of expert predictions for X.
    :param batches: output of generate_batches().
    :param capacity: output of generate_capacity().
    :param output_dir: directory to save to.
    """
    data = X.copy()
    data['batch'] = -1
    data['assignment'] = 'N/A'
    data['decision'] = 'N/A'
    print(f'Sampling expert predictions for the combination {batch_scheme},{capacity_scheme}')
    for i in np.arange(1,batches['batch'].max()+1):
        c = capacity.loc[i,:].copy()
        cases = data.loc[batches.loc[batches['batch'] == i].index,:]
        human_cap = c.sum()
        to_review = cases.sample(frac = 1, replace = False, random_state = 42)
        to_review = to_review.iloc[:human_cap,:]    
        experts = expert_pred.columns.to_list()
        data.loc[cases.index, 'batch'] = i
        for expert in experts:
            if c[expert]>0:
                set_for_expert = to_review.sample(n = c[expert], random_state = 42)
                data.loc[set_for_expert.index, 'assignment'] = expert
                data.loc[set_for_expert.index, 'decision'] = expert_pred.loc[set_for_expert.index, expert]
                to_review = to_review.drop(index = set_for_expert.index)

    assgn_n_dec = data[['assignment', 'decision']]

    if output_dir is not None:
        assgn_n_dec.to_parquet(output_dir + 'assignments_and_decisions.parquet')

    return assgn_n_dec

# TRAIN --------------------------------------------------------------------------------------------

train_envs = generate_environments(
    df=train,
    batch_cfg=cfg['environments_train']['batch'],
    capacity_cfg=cfg['environments_train']['capacity'],
    output_dir=f'{cfg["destination_path_train"]}/',
)

for (batch_scheme, capacity_scheme), (train_batches, train_capacity) in train_envs.items():
    env_assignment_and_pred = generate_predictions(
        X=train,
        expert_pred=train_expert_pred,
        batches=train_batches,
        capacity=train_capacity
    )
    env_train = (
        train
        .merge(train_batches, left_index=True, right_index=True)
        .merge(env_assignment_and_pred, left_index=True, right_index=True)
    )
    env_train.to_parquet(
        f'{cfg["destination_path_train"]}/{batch_scheme}#{capacity_scheme}/train.parquet'
    )


