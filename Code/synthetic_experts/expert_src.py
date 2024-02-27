###
from abc import ABC, abstractmethod
import numpy as np
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import math

def sig(x):
    return 1/(1+np.exp(-x))

def output(data, model, init_score):
    return sig(model.predict(data,raw_score=True) + init_score)

def sigmoid(x):
    if x<-15:
        return 0
    elif x>15:
        return 1
    else:
        return 1/(1+math.exp(-x))

def inv_sigmoid(x):
    return math.log(x/(1-x))

def invert_labels_with_probabilities(labels_arr, p_arr, seed):
    rng = np.random.default_rng(seed=seed)
    mask = rng.binomial(n=1, p=p_arr).astype(bool)

    new_labels = labels_arr.copy()
    new_labels[mask] = np.abs(new_labels[mask] - 1)  # inverts labels

    return new_labels



class AbstractExpert(ABC):

    @abstractmethod
    def predict(self):
        pass


class SigmoidExpert(AbstractExpert):

    def __init__(
            self,
            fnr_target: float, fpr_target: float,
            features_w_mean: float,
            features_w_std: float,
            protected_w: float,
            score_w: float,
            alpha: float,
            fpr_noise: float, fnr_noise: float, theta: float,
            features_dict = None,
            seed=42,
    ):
        self.fnr = fnr_target
        self.fpr = fpr_target
        self.features_w_mean = features_w_mean
        self.features_w_std = features_w_std
        self.alpha = alpha
        self.fpr_noise = fpr_noise
        self.fnr_noise = fnr_noise
        self.protected_w= protected_w
        self.score_w = score_w
        self.theta = theta
        self.features_dict = features_dict
        self.seed = seed

        # params set by fit
        self.fnr_beta = None
        self.w = None
        self.fpr_beta = None
        self.error_prob = pd.DataFrame(-1, index=np.arange(1000000), columns=['p_of_fp', 'p_of_fn'])


    def fit(self, X, y, score_col, protected_col):

        self.fnr_beta = self.fnr
        self.fpr_beta = self.fpr
        
        if self.features_dict is None:
            np.random.seed(self.seed)
            spike = np.random.binomial(n=1, p=self.theta, size = (X.shape[1],))
            slab = np.random.normal(loc=self.features_w_mean, scale=self.features_w_std, size=(X.shape[1],))
            self.w = np.multiply(spike,slab)

            if self.score_w is not None:
                self.w[X.columns.get_loc(score_col)] = self.score_w
            if self.protected_w is not None:
                self.w[X.columns.get_loc(protected_col)] = self.protected_w
        else:
            np.random.seed(self.seed)
            self.w = np.zeros(X.shape[1])
            for feature in self.features_dict.keys():
                self.w[X.columns.get_loc(feature)] = np.random.normal(loc = self.features_dict[feature][0], scale = self.features_dict[feature][1])

        tolerance = 0.00001
        fpr_a = -200
        fpr_b = 200

        fnr_a = -200
        fnr_b = 200


        self.fpr_beta = fpr_a
        self.calc_probs_fp(X=X, y=y)
        error_prob_fit = self.error_prob.loc[X.index,:]
        fp_mean_a = error_prob_fit.loc[y == 0, 'p_of_fp'].mean()

        self.fpr_beta = fpr_b
        self.calc_probs_fp(X=X, y=y)
        error_prob_fit = self.error_prob.loc[X.index,:]
        fp_mean_b = error_prob_fit.loc[y == 0, 'p_of_fp'].mean()

        assert((fp_mean_a - self.fpr) * (fp_mean_b - self.fpr) < 0)

        self.fpr_beta = (fpr_b + fpr_a)/2
        self.calc_probs_fp(X=X, y=y)
        error_prob_fit = self.error_prob.loc[X.index,:]
        fp_mean = error_prob_fit.loc[y == 0, 'p_of_fp'].mean()
        

        while np.abs(fp_mean - self.fpr) > tolerance:
            if (fp_mean_a - self.fpr) * (fp_mean - self.fpr) < 0:
                fpr_b = self.fpr_beta
                fp_mean_b = fp_mean
                self.fpr_beta = (fpr_b + fpr_a)/2
                self.calc_probs_fp(X=X, y=y)
                
                error_prob_fit = self.error_prob.loc[X.index,:]
                fp_mean = error_prob_fit.loc[y == 0, 'p_of_fp'].mean()

            elif (fp_mean_b - self.fpr) * (fp_mean -self.fpr) < 0:
                fpr_a = self.fpr_beta
                fp_mean_a = fp_mean
                self.fpr_beta = (fpr_b + fpr_a)/2
                self.calc_probs_fp(X=X, y=y)
                error_prob_fit = self.error_prob.loc[X.index,:]
                fp_mean = error_prob_fit.loc[y == 0, 'p_of_fp'].mean()

        self.fnr_beta = fnr_a
        self.calc_probs_fn(X=X, y=y)
        error_prob_fit = self.error_prob.loc[X.index,:]
        fn_mean_a = error_prob_fit.loc[y == 1, 'p_of_fn'].mean()

        self.fnr_beta = fnr_b
        self.calc_probs_fn(X=X, y=y)
        error_prob_fit = self.error_prob.loc[X.index,:]
        fn_mean_b = error_prob_fit.loc[y == 1, 'p_of_fn'].mean()

        assert((fn_mean_a - self.fnr) * (fn_mean_b - self.fnr) < 0)

        self.fnr_beta = (fnr_b + fnr_a)/2
        self.calc_probs_fn(X=X, y=y)
        error_prob_fit = self.error_prob.loc[X.index,:]
        fn_mean = error_prob_fit.loc[y == 1, 'p_of_fn'].mean()
        

        while np.abs(fn_mean - self.fnr) > tolerance:
            if (fn_mean_a - self.fnr) * (fn_mean - self.fnr) < 0:
                fnr_b = self.fnr_beta
                fn_mean_b = fn_mean
                self.fnr_beta = (fnr_b + fnr_a)/2
                self.calc_probs_fn(X=X, y=y)
                error_prob_fit = self.error_prob.loc[X.index,:]
                fn_mean = error_prob_fit.loc[y == 1, 'p_of_fn'].mean()

            elif (fn_mean_b - self.fnr) * (fn_mean -self.fnr) < 0:
                fnr_a = self.fnr_beta
                fn_mean_a = fn_mean
                self.fnr_beta = (fnr_b + fnr_a)/2
                self.calc_probs_fn(X=X, y=y)
                error_prob_fit = self.error_prob.loc[X.index,:]
                fn_mean = error_prob_fit.loc[y == 1, 'p_of_fn'].mean()


    def calc_probs_fp(self, X, y, **kwargs):  # kwargs not used (compatibility purposes)
        if self.w is None:
            raise ValueError('Synthetic expert must be .fit() to the data.')
        
        weights = self.w
        
        probability_of_fp = (y == 0) * (
            self.fpr_beta + (self.alpha*(X * weights/(np.linalg.norm(weights))).sum(axis=1)
            )).apply(sigmoid)
        self.error_prob.loc[X.index,'p_of_fp'] = probability_of_fp
        
    def calc_probs_fn(self, X, y, **kwargs):  # kwargs not used (compatibility purposes)
        if self.w is None:
            raise ValueError('Synthetic expert must be .fit() to the data.')
        
        weights = self.w

        probability_of_fn = (y == 1) * (
            self.fnr_beta - (self.alpha*(X * weights/(np.linalg.norm(weights))).sum(axis=1)
            )).apply(sigmoid)
    
        self.error_prob.loc[X.index,'p_of_fn'] = probability_of_fn

    def predict(self, X, y, **kwargs):  # kwargs not used (compatibility purposes)
        if self.w is None:
            raise ValueError('Synthetic expert must be .fit() to the data.')
        
        weights = self.w
        """
        probability_of_fn = (y == 1) * (
            inv_sigmoid(self.fnr_beta) + (self.alpha*(X * weights/(np.linalg.norm(weights))).sum(axis=1)
            )).apply(sigmoid)
        
        probability_of_fp = (y == 0) * (
            inv_sigmoid(self.fpr_beta) - (self.alpha*(X * weights/(np.linalg.norm(weights))).sum(axis=1)
            )).apply(sigmoid)
        
        """

        probability_of_fn = (y == 1) * (
            self.fnr_beta - (self.alpha*(X * weights/(np.linalg.norm(weights))).sum(axis=1)
            )).apply(sigmoid)
        
        probability_of_fp = (y == 0) * (
            self.fpr_beta + (self.alpha*(X * weights/(np.linalg.norm(weights))).sum(axis=1)
            )).apply(sigmoid)
  
        probability_of_error = probability_of_fn + probability_of_fp

        decisions = invert_labels_with_probabilities(
            labels_arr=y,
            p_arr=probability_of_error,
            seed=self.seed
        )

        #error_df = pd.DataFrame()
        #error_df['p_of_fn'] = probability_of_fn
        #error_df['p_of_fp'] = probability_of_fp

   
        self.error_prob.loc[X.index,'p_of_fn'] = probability_of_fn
        self.error_prob.loc[X.index,'p_of_fp'] = probability_of_fp
        return decisions
    

class ExpertTeam(dict):

    def __init__(self, experts=None):
        if experts is None:
            experts = dict()
        self.experts = self._convert_to_dict(experts)
        super().__init__(experts)

    def fit(self, **kwargs):
        i = 0
        for _, expert_obj in self.items():
            if i>=0:
                print(f'fitting expert n: {i}')
                i+=1
                expert_obj.fit(**kwargs)
            else:
                i+=1

    def predict(
            self,
            index,
            predict_kwargs: dict,
            long_format=False, assignment_col=None, decision_col=None,
    ):
        predictions_dict = dict()
        for expert_id, expert in self.items():
            predictions_dict[expert_id] = expert.predict(**predict_kwargs[type(expert)])
        predictions_df = pd.DataFrame(predictions_dict, index=index, columns=list(self.keys()))

        if long_format:
            predictions_df = predictions_df.reset_index()
            predictions_df = predictions_df.melt(
                id_vars=index.name,
                var_name=assignment_col,
                value_name=decision_col
            )

        return predictions_df

    def query(self, index, assignments, **kwargs):
        predictions = self.predict(index, **kwargs)
        mask = np.array(
            [assignments == e for e in predictions.columns]
        ).T
        queried_decisions = pd.Series(
            predictions.values[mask],
            index=index
        )

        return queried_decisions

    @staticmethod
    def _convert_to_dict(experts) -> dict:
        if isinstance(experts, (list, tuple)):
            experts_dict = {i: experts[i] for i in range(len(experts))}
        elif isinstance(experts, dict):
            experts_dict = experts
        else:
            raise ValueError('experts must be either a list, a tuple, or, preferibly, a dict.')

        return experts_dict
