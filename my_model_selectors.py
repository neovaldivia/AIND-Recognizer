import math
import statistics
import warnings
import logging

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        # TODO implement model selection based on *lowest* BIC scores
        # TODO implement model selection based on *lowest* BIC scores
        n_components = range(self.min_n_components,self.max_n_components+1)
        best_n = self.random_state
        best_BIC = float('inf')
        best_model = None
        
        for n in n_components:
            try:
                model = GaussianHMM(n,n_iter=1000).fit(self.X,self.lengths)
                logL = model.score(self.X,self.lengths)
                
                #p=all parameters to learn, features: N=len(self.X[0])
                p = n * n + 2 * n * len(self.X[0]) - 1
                BIC = -2 * logL + 2 * p * math.log(len(self.X))                                
                
            except:
                BIC = float('inf')       
                
            if BIC<best_BIC:
                best_BIC=BIC
                best_model = model
                    
        return best_model


class SelectorDIC(ModelSelector):
    
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # TODO implement model selection based on DIC scores
        #DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
                #where
                #P(X(i)) = original_prob = logL for X(i)
                #M = #classes=#words
                
        n_components = range(self.min_n_components,self.max_n_components+1)
        best_DIC = float('-inf')
        best_model = None
        for n in n_components:
            try:       
                model = GaussianHMM(n,n_iter=1000).fit(self.X,self.lengths)
                original_prob = model.score(self.X,self.lengths)
                count = 0
                sum_prob_others = 0.0
                
                for word in self.words:
                    if word==self.this_word:
                        continue
                        
                    X_other, lengths_other = self.hwords[word]  
                   
                    logL = model.score(X_other,lengths_other)
                    sum_prob_others += logL  
                    count+=1
                    
                avg_prob_others = sum_prob_others/float(count)
                DIC = original_prob - avg_prob_others
            
            except:
                DIC = float('-inf')
                
            if DIC>best_DIC:
                best_DIC=DIC
                best_model = model
                 
        return best_model
    
class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection using CV 
        # TODO implement model selection using CV 
        best_score = float('-inf')
        best_model = None
        score_sum = 0.0
        split_method = KFold(n_splits=2)
        n_components = range(self.min_n_components, self.max_n_components+1)
        
        for n in n_components:
            try:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    X_train,lengths_train = combine_sequences(cv_train_idx,self.sequences)            
                    X_test,lengths_test = combine_sequences(cv_test_idx,self.sequences)
                    model = GaussianHMM(n,n_iter=1000).fit(X_train,lengths_train)
                    logL = model.score(X_test,lengths_test)
                    score_sum += logL
                score_avg = score_sum/(n-self.min_n_components+1)
            
            except:
                score_avg = float('-inf')
                
            if score_avg > best_score:
                best_score=score_avg
                best_model = model
            
        return best_model
        
    
