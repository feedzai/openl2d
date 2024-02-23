# %%
import os
import random
import numpy as np
import pandas as pd
import yaml

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

with open('cfg.yaml', 'r') as infile:
    cfg = yaml.safe_load(infile)

np.random.seed(cfg['random_seed'])
random.seed(cfg['random_seed'])

with open('../alert_data/dataset_cfg.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)

# DATA LOADING -------------------------------------------------------------------------------------
data = pd.read_parquet(f'../../FiFAR/alert_data/processed_data/alerts.parquet')

with open(f'../../FiFAR/synthetic_experts/expert_ids.yaml', 'r') as infile:
    EXPERT_IDS = yaml.safe_load(infile)

EXPERT_CATS = EXPERT_IDS['human_ids']

train = data.loc[(data["month"] > 2) & (data["month"] < 7)]
train = cat_checker(train, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

# EXPERTS ------------------------------------------------------------------------------------------
# produced in experts/experts_generation.py
experts_pred = pd.read_parquet(f'../../FiFAR/synthetic_experts/expert_predictions.parquet')
train_expert_pred = experts_pred.loc[train.index, ]
# BATCH & CAPACITY ---------------------------------------------------------------------------------
def generate_batches(df, batch_properties: dict, months: pd.Series) -> pd.DataFrame:
    """
    Generates a pandas dataframe indicating the (serial) number of the batch each instance belongs to.
    Batches do not crossover from one month to the other.
    :param batch_properties: dictionary containing size key-value pair (see cfg.yaml).
    :param months: pandas series indicating the month of each instance.
    """
    batches_months_list = list()
    last_batch_ix = 0
    for m in months.unique():
        df_m = df[months == m]
        df_m = df_m.sample(frac = 1, random_state = batch_properties['seed'])
        
        m_batches = pd.DataFrame(
            [int(i / batch_properties['size']) + last_batch_ix + 1 for i in range(len(df_m))],
            index=df_m.index,
            columns=['batch'],
        )
        batches_months_list.append(m_batches)
        last_batch_ix = int(m_batches.max())

    batches = pd.concat(batches_months_list)

    return batches

def generate_capacity_single_batch(batch_size: int, properties: dict, human_ids: list) -> dict:
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
            (batch_size) /
            len(human_ids)
        )
        unc_human_capacities = np.full(shape=(len(human_ids),), fill_value=humans_capacity_value)

    elif properties['distribution'] == 'variable':  # capacity follows a random Gaussian
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
        random.seed(properties['n_experts_seed'])
        absent_humans_ix = random.sample(  # without replacement
            available_humans_ix,
            k=len(available_humans_ix) - int(properties['n_experts']),
        )
        unc_human_capacities[absent_humans_ix] = 0

        unassigned = (batch_size  - sum(unc_human_capacities))
        available_humans_ix = [ix for ix in available_humans_ix if ix not in absent_humans_ix]
        unc_human_capacities = unc_human_capacities.astype(float)
        unc_human_capacities[available_humans_ix] *= (1 + unassigned / sum(unc_human_capacities))

    # convert to integer and adjust for rounding errors
    human_capacities = np.floor(unc_human_capacities).astype(int)
    unassigned = int(batch_size - sum(human_capacities))
    assert unassigned < len(human_ids)
    to_add_to = random.sample(available_humans_ix, k=unassigned)
    human_capacities[to_add_to] += 1

    capacity_dict.update(**{
        human_ids[ix]: int(human_capacities[ix])
        for ix in range(len(human_ids))
    })

    assert sum(capacity_dict.values()) == batch_size

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
            batches_df = generate_batches(
                df=df,
                batch_properties=batch_properties,
                months=df["month"]
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
    data['assignment'] = 'model#0'
    data['decision'] = data['model_score']
    for i in np.arange(1,batches['batch'].max()+1):
        print(f'batch {i}')
        c = capacity.loc[i,:].copy()
        print(c)
        cases = data.loc[batches.loc[batches['batch'] == i].index,:]
        human_cap = c.sum()
        print(human_cap)
        to_review = cases.sort_values(by = 'model_score', ascending = False)
        to_review = to_review.iloc[:human_cap,:]    
        experts = expert_pred.columns.to_list()
        data.loc[cases.index, 'batch'] = i
        
        for ix in to_review.index:
            done = 0
            while (done != 1):
                choice  = random.choice(experts)
                if c[choice]>0:
                    c[choice] -= 1
                    data.loc[ix, 'assignment'] = choice
                    data.loc[ix, 'decision'] = expert_pred.loc[ix, choice]
                    done = 1
                else:
                    experts.remove(choice)
                if len(choice) == 0:
                    done = 1

    assgn_n_dec = data[['assignment', 'decision']]

    if output_dir is not None:
        assgn_n_dec.to_parquet(output_dir + 'assignments_and_decisions.parquet')

    return assgn_n_dec

# TRAIN --------------------------------------------------------------------------------------------

train_envs = generate_environments(
    df=train,
    batch_cfg=cfg['environments_train']['batch'],
    capacity_cfg=cfg['environments_train']['capacity'],
    output_dir=f'../../FiFAR/testbed/train_alert/',
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
        f'../../FiFAR/testbed/train_alert/{batch_scheme}#{capacity_scheme}/train.parquet'
    )


