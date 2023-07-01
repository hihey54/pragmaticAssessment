import sys
import os
import pandas as pd
import time
import datetime as dt
import numpy as np
import ipaddress
from pathlib import Path
import json
import matplotlib
import math
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score, train_test_split, learning_curve, \
    cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, \
    make_scorer, recall_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler, minmax_scale, scale
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.linear_model import *
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import *
from sklearn.neural_network import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.svm import *
from mlxtend.classifier import StackingClassifier

pd.options.mode.use_inf_as_na = False
pd.options.mode.chained_assignment = None  # default='warn'


class Result:
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    
    def __init__(self, true, pred, time, infer_time, pos_label=1):
        self.time = time
        self.infer_time = infer_time
        self.rec = 0
        self.f1 = 0
        self.fpr = 0
        self.acc = 0
        self.acc_multi = 0
        if len(pd.Series(true).unique())==2: #binary classification
            if len(pd.Series(pred).unique())==2:
                self.bin_results(true, pred, pos_label)
            else:
                self.multi_results(true, pred)
            
        else: #multi class or different
            self.multi_results(true, pred)
        
    def multi_results(self, true, pred):
        self.ctab = pd.crosstab(true, pred, rownames=['True'], colnames=['Pred'])           
        self.acc_multi = accuracy_score(true, pred, normalize=True, sample_weight=None)
    
    def bin_results(self, true, pred, pos_label = 1):
        self.ctab_bin = pd.crosstab(true, pred, rownames=['True'], colnames=['Pred'])
        self.acc = accuracy_score(true, pred, normalize=True, sample_weight=None)
        try:
            self.rec = recall_score(true, pred, zero_division=0, pos_label=pos_label)
            self.f1 = f1_score(true, pred, zero_division=0, pos_label=pos_label)
            self.tnr = recall_score(true, pred, zero_division=0, pos_label=0)
        except:
            self.rec = 0
            self.f1 = 0
            self.tnr = self.acc
        self.fpr = 1-self.tnr
        
        
    

    
def compute_results(true, pred, time):
    import pandas as pd
    ctab = pd.crosstab(truth, pred, rownames=['True'], colnames=['Pred'])
    return Result(ctab, time)

        
def develop_clf(train, test, features, clf_name='classifier', label='Nature', clf_type = 'rf', verbose=False):
    # Function that trains and tests a classifier, returning the results
    import pandas as pd
    import time
    train_y = train[label]
    test_y = test[label]
    
    clf = choose_clf(clf_type)
    
    start_time = time.time()
    if verbose:
        print("Training and testing {}...".format(clf_name), end="", flush=True)
    clf.fit(train[features], train_y)
    clf_time = time.time() - start_time
    clf_pred = clf.predict(test[features])
    clf_infer_time = time.time() - clf_time - start_time
    clf_result = Result(test_y, clf_pred, clf_time, clf_infer_time)
    if verbose:
        print("...done! Training time: {:3f}s\tInference time: {:3f}s".format(clf_time, clf_infer_time))
    return clf, clf_pred, clf_result

def evaluate_clf(clf, test, features, clf_name='classifier', label='Nature', time=None, verbose=False):
    import pandas as pd
    import time
    if verbose==True:
        print("Testing {}...".format(clf_name))
    start_time = time.time()
    clf_pred = clf.predict(test[features])
    infer_time = time.time() - start_time
    clf_result = Result(test[label], clf_pred, time, infer_time)
    if verbose:
        print("...done! \tInference time: {:3f}s".format(infer_time))
    return clf_pred, clf_result, infer_time
    
def train_clf(train_data, features, clf_name='classifier', label='Nature', verbose=False):
    import pandas as pd
    import time
    train_y = train_data[label]
    clf = choose_clf(clf_type)
    
    start_time = time.time()
    if verbose:
        print("Training {}...".format(clf_name), end="", flush=True)
    clf.fit(train_data[features], train_y)
    clf_time = time.time() - start_time
    if verbose:
        print("...done! Training time: {}".format(clf_time))
    return clf, clf_time
    
    
    
    
    
def choose_clf(clf_type):
    if clf_type == 'rf':
        clf = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None, min_samples_split=2, 
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                                 max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, 
                                 n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None, 
                                 ccp_alpha=0.0, max_samples=None)
    elif clf_type == 'lr':
        clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
                                intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, 
                                multi_class='auto', verbose=0, warm_start=False, n_jobs=-1, l1_ratio=None)
    elif clf_type == 'hgb':
        clf = HistGradientBoostingClassifier(loss='auto', learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, 
                                    min_samples_leaf=20, l2_regularization=0.0, max_bins=255,
                                    monotonic_cst=None, warm_start=False, early_stopping='auto', scoring='loss', 
                                    validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None)
    elif clf_type == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.0001, 
                            batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
                            max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
                            warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
                            validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                            n_iter_no_change=20, max_fun=15000)
    elif clf_type == 'knn':
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, 
                                   metric='minkowski', metric_params=None, n_jobs=-1)
    elif clf_type == 'dt':
        clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
                                     min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                     random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                     class_weight=None, ccp_alpha=0.0)
    elif clf_type == 'svm':
        clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)
    
    return clf