""" 
Ensemble-Framework: An collection of functions that implement Ensemble Learning techniques such as Voting and Boosting.
"""

import numpy as np
import itertools
from collections import defaultdict
import time
import random
import HMM_Framework


random_state = 22
random.seed(22)

### Boosting Ensemble - operates during training ###
def adaboost():
    print("TODO")

###                                              ###

### Voting Ensembles - operate at the end of the entire process ###
def ensemble_run(cross_val_prediction_matrix, mapping, golden_truth, mode, weights=None, use_log_prob=True, detailed=False):
    """
    After training multiple models using the HMM framework we can add the following objects to a list: hmm.cross_val_prediction_matrix
                                                                                                       hmm.ensemble_stored["Mapping"]
                                                                                                       hmm.ensemble_stored["Curr_Cross_Val_Golden_Truth"]
                                                                                                       where list[current_model][current_cross_val_fold]
    Given those objects as parameters, calculates a voting ensemble of all the models.

    mode:   
            sum: Weighted Voting Ensemble on probability matrices with a sum operation, also known as Soft Voting in literature
            product: Weighted Voting Ensemble on probability matrices with a product operation, also known as Soft Voting in literature
            borda: Borda count, a single-winner ranking method
    """
    model_count = len(cross_val_prediction_matrix)

    # Perform certain input validation checks
    if model_count < 2:
        print("\n--Warning: You have inputted less then two models, the Ensemble process is pointless.")
    if weights != None:
        if len(weights) != model_count:
            raise ValueError("you must give as many weights as there are models or simply 'None'.")
    else:
        weights = [1.0/model_count] * model_count

    try:
        if isinstance(cross_val_prediction_matrix[0][0], np.ndarray) != True:
            raise ValueError("the format of the prediction matrix seems to be wrong, are you certain there were multiple cross validation folds?") 
        else:
            cross_val_folds = len(cross_val_prediction_matrix[0])
    except TypeError:
        raise ValueError("the format of the prediction matrix seems to be wrong, are you certain there were multiple cross validation folds?")        

    for curr_fold in range(cross_val_folds):
        for model in range(model_count-1):
            if np.array_equal(golden_truth[model][curr_fold], golden_truth[model+1][curr_fold]) != True:  # or we could have used python's 'all()' function
                raise ValueError("the golden truth labels across models and across cross validation folds are not identical.")
            if mapping[model][curr_fold] != mapping[model+1][curr_fold]:
                raise ValueError("the mapping across models and across cross validation folds is not identical.")
    golden_truth = golden_truth[0]  # Everything is OK, we only need to keep the golden truth and mapping of any one model   
    mapping = mapping[0]        
    #

    if use_log_prob == False:
        for curr_fold in range(cross_val_folds):
            for model in range(model_count-1):
                cross_val_prediction_matrix[model][curr_fold] = np.exp(cross_val_prediction_matrix[model][curr_fold])

    # Create a HMM object just to use the 'result_metrics' function
    dummy_object = HMM_Framework.HMM_Framework()
    dummy_object.selected_model = "Ensemble of HMMs"
    dummy_object.k_fold = cross_val_folds

    # Run the Ensemble
    for curr_fold in range(cross_val_folds):
        time_counter = time.time()

        if mode != "borda":
            for model in range(model_count):
                    if model == 0:
                        ensemble_matrix = weights[0]*cross_val_prediction_matrix[0][curr_fold]
                    else:
                        if mode == "sum":
                            ensemble_matrix += weights[model]*cross_val_prediction_matrix[model][curr_fold]
                        elif mode == "product":
                            ensemble_matrix *= weights[model]*cross_val_prediction_matrix[model][curr_fold]

            prediction = []
            total_models = len(mapping[curr_fold])

            # Old Way
            #indices = np.argmax(ensemble_matrix, axis=1)  
                      
            for instance in range(ensemble_matrix.shape[0]):
                curr_instance = ensemble_matrix[instance, :]
                # Comparison
                if len(set(curr_instance)) == 1:  # Ensure that we don't have n equal predictions, where argmax wouldn't work
                    index = random.randint(0, total_models - 1)
                else:
                    index = np.argmax(curr_instance)

                prediction.append(mapping[curr_fold][index])

        else:
            prediction = borda_count(curr_fold, model_count, cross_val_prediction_matrix, mapping)
                   
        dummy_object.result_metrics(golden_truth[curr_fold], prediction, time_counter)
    
    dummy_object.print_average_results(decimals=3)
    dummy_object.print_best_results(detailed=detailed, decimals=3) 

def borda_count(curr_fold, model_count, cross_val_prediction_matrix, mapping):
    """
    Implementation of Borda count, a single-winner ranking method.
    """
    instance_count = cross_val_prediction_matrix[0][curr_fold].shape[0]
    prediction = []

    for instance in range(instance_count):
        ballots = []        
        for model in range(model_count):
            sorted_indices = list(np.argsort(cross_val_prediction_matrix[model][curr_fold][instance, :]))
            sorted_indices.reverse()  # Reverse
            temp = ''
            for j in range(len(sorted_indices)):  # j is the current index and sorted_indices[j] is the current label
                if j > 0:
                    if sorted_indices[j-1] == sorted_indices[j]:
                        temp += '='
                    else:
                        temp += '>'
                temp += mapping[curr_fold][sorted_indices[j]]
            ballots.append(temp)
            # ballots = ['A>B>C>D>E',
            #           'A>B>C>D=E',
            #           'A>B=C>D>E', 
            #           'B>A>C>D',
            #            ]        

        prediction.append(max(_borda_count_tally(ballots), key=_borda_count_tally(ballots).get))  # Maximum value is not always on the leftmost position of the Dict so we need to find it

    return(prediction)

def _borda_count_main(ballot):
    n = len([c for c in ballot if c.isalpha()]) - 1
    score = itertools.count(n, step = -1)
    result = {}
    for group in [item.split('=') for item in ballot.split('>')]:
        s = sum(next(score) for item in group)/float(len(group))
        for pref in group:
            result[pref] = s
    return result

def _borda_count_tally(ballots):
    result = defaultdict(int)
    for ballot in ballots:
        for pref,score in _borda_count_main(ballot).items():
            result[pref]+=score
    result = dict(result)
    return result
###                                                             ###

