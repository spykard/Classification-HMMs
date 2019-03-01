''' 
Sentiment Analysis: Text Classification using Hidden Markov Models
'''

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import itertools
import numpy as np
import pandas as pd
from math import log
import random
from collections import defaultdict
import time as my_time  # Required to avoid some sort of conflict with pomegranate

from pomegranate import HiddenMarkovModel
from pomegranate import DiscreteDistribution
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from nltk import ngrams


cross_validation_best = [0.0, 0.0, "", [], [], 0.0]           # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
cross_validation_best_ensemble = [0.0, 0.0, "", [], [], 0.0]  # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
cross_validation_all = defaultdict(list)                      # {Name: (Accuracy, F1-score), (Accuracy, F1-score), ...}
cross_validation_average = defaultdict(list)                  # {Name: (Avg(Accuracy), Avg(F1-score)), ...}
time_complexity_average = defaultdict(list)                   # {Name: [Avg(Train+Test_Time)]


def Run_Preprocessing(dataset_name):
    '''    Dataset Dependant Preprocessing    '''

    # 1. Load the Dataset
    # data = [["" for j in range(3)] for i in range(294)]
    data = ["" for i in range(294)]
    sequences = [[] for i in range(294)]
    labels = ["" for i in range(294)]
    count = 0
    with open('./Datasets/Finegrained/finegrained.txt', 'r') as file:
        for line in file:
            if len(line.split("_")) == 3:
                labels[count] = line.split("_")[1]
            elif len(line.strip()) == 0:
                count += 1
            else:
                temp = [x.strip() for x in line.split("\t")]
                if len(temp[1]) > 1:
                    # "nr" label is ignored
                    if temp[0] in ["neg", "neu", "pos"]:
                        sequences[count].append(temp[0])              

                    data[count] += temp[1]

    print("--Processed", count+1, "documents", "\n--Dataset Name:", dataset_name)

    df_dataset = pd.DataFrame({'Labels': labels, 'Data': data, 'Sequences': sequences})

    # 2. Remove empty instances from DataFrame, actually affects accuracy
    emptyCells = df_dataset.loc[df_dataset.loc[:,'Sequences'].map(len) < 1].index.values
    df_dataset = df_dataset.drop(emptyCells, axis=0).reset_index(drop=True)  # Reset_Index to make the row numbers be consecutive again

    # 3. Balance the Dataset by Undersampling
    # mask = df_dataset.loc[:,'Labels'] == "No"
    # df_dataset_to_undersample = df_dataset[mask].sample(n=1718, random_state=22)
    # df_dataset = df_dataset[~mask]
    # df_dataset = pd.concat([df_dataset, df_dataset_to_undersample], ignore_index=True)
    # df_dataset = df_dataset.sample(frac=1, random_state=22).reset_index(drop=True)

    # 4. Shuffle the Datasets, it seems to be too perfectly ordered
    # df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)

    return df_dataset


def HMM_NthOrder_Supervised(pos_data_train, neg_data_train, neu_data_train, data_test, labels_test, documentSentiments, targetnames, n_jobs, algorithm, graph_print_enable, silent_enable, silent_enable_2, n_order):
    '''    Create a Hidden Markov Model of n-th Order, trained in a Supervised manner    '''
    '''    Input:     
                    data_train, data_test, labels_train, labels_test: Data and their labels
                    documentSentiments: A list of Unique Sentiments which are the Hidden States
                    targetnames: Optional list of labels to display on the testing/evaluation phase
                    n_jobs: The number of jobs to run in parallel. Value of 1 means use 1 processor and -1 means use all processors
                    algorithm: Prediction Algorithm, can be either:
                                Viterbi ('viterbi') or Maximum a posteriori ('map')
                    graph_print_enable: Enables the plotting the graph of the Hidden Markov Model
                    silent_enable: If set to 1 disables printing the entire matrix of observation probabilities
                    silent_enable_2: If set to 1 disables printing metrics and confusion matrix on every single model that is built
                    n_order: The order n of the Hidden Markov Model that is about to be built '''
    '''    Output:  
                    If Maximum a posteriori is chosen, returns the Log probability Matrix of Predictions,
                    else returns nothing, since we will not be performing an Ensemble on Viterbi '''
    '''    Note:
                    Mathematical Foundation: Even though the hidden state labels are constrained to remain the same across each sequence/instance
                    and one could argue that the model doesn't learn anything about transitions between hidden states, it still appears to learn transitions fine '''

    time_counter = my_time.time()
    random.seed(22)

    set_order = 2

    pos_data_train = pos_data_train.values.tolist()
    neg_data_train = neg_data_train.values.tolist()
    neu_data_train = neu_data_train.values.tolist()
    data_test = data_test.values.tolist()

    # The sequences are stored as a List in the Dataframe, time to transform them to the correct form
    # No need for dummy states because we have the same labels as observations, it's irelevant
    # Pos    
    second_order_labels = []    
    for seq in pos_data_train:
        tempp = []
        # wtf bug
        current_seq = seq[0]

        if len(current_seq) > 0:  # else remains empty
            for i in range(set_order-1, len(current_seq)):
                # concatenate the names of 2 states
                tempp.append(current_seq[i-1] + current_seq[i])

        second_order_labels.append(tempp)
    pos_data = second_order_labels  # Convert the name

    # Neg    
    second_order_labels = []    
    for seq in neg_data_train:
        tempp = []
        # wtf bug
        current_seq = seq[0]

        if len(current_seq) > 0:  # else remains empty
            for i in range(set_order-1, len(current_seq)):
                # concatenate the names of 2 states
                tempp.append(current_seq[i-1] + current_seq[i])

        second_order_labels.append(tempp)
    neg_data = second_order_labels  # Convert the name

    # Neu
    second_order_labels = []    
    for seq in neu_data_train:
        tempp = []
        # wtf bug
        current_seq = seq[0]

        if len(current_seq) > 0:  # else remains empty
            for i in range(set_order-1, len(current_seq)):
                # concatenate the names of 2 states
                tempp.append(current_seq[i-1] + current_seq[i])

        second_order_labels.append(tempp)
    neu_data = second_order_labels  # Convert the name

    # Test
    second_order_labels = []    
    for seq in data_test:
        tempp = []
        # no bug here

        if len(current_seq) > 0:  # else remains empty
            for i in range(set_order-1, len(seq)):
                # concatenate the names of 2 states
                tempp.append(seq[i-1] + seq[i])

        second_order_labels.append(tempp)
    test_data = second_order_labels  # Convert the name

    # print(test_data)
    # quit()

    # ### Supervised Training
    # # In this case we need to find out which State corresponds to each label (pos/neg/neu) before training  
    # labels_supervised = list()
    # for i, x in enumerate(labels_train):
    #     getlength = len(data_train_transformed[i])
    #     state_name = "s" + str(documentSentiments.index(x))
    #     labels_supervised.append([state_name] * getlength)

    # state_names = list()
    # for i in range(0, len(documentSentiments)):
    #     state_names.append("s" + str(i))
   
        # Build Pos Class HMM - !!! state_names should be in alphabetical order
    hmm_supervised_pos = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=9, X=pos_data, labels=pos_data, n_jobs=1, emission_pseudocount=0.99, verbose=True, state_names=['negneg','negneu','negpos','neuneg','neuneu','neupos','posneg','posneu','pospos'])
    print("NEXT HMM")
    # Build Neg Class HMM - !!! state_names should be in alphabetical order
    hmm_supervised_neg = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=9, X=neg_data, labels=neg_data, n_jobs=1, emission_pseudocount=0.99, verbose=True, state_names=['negneg','negneu','negpos','neuneg','neuneu','neupos','posneg','posneu','pospos'])
    hmm_supervised_neu = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=9, X=neu_data, labels=neu_data, n_jobs=1, emission_pseudocount=0.99, verbose=True, state_names=['negneg','negneu','negpos','neuneg','neuneu','neupos','posneg','posneu','pospos'])

    
    # Note: Algorithm used is Baum-Welch
    ###

    transition_proba_matrix_pos = hmm_supervised_pos.dense_transition_matrix()
    transition_proba_matrix_neg = hmm_supervised_neg.dense_transition_matrix()
    transition_proba_matrix_neu = hmm_supervised_neu.dense_transition_matrix()

    # ### Print information about the Hidden Markov Model such as the probability matrix and the hidden states
    #print(hmm_supervised_pos)
    # ###

    # ### Plot the Hidden Markov Model Graph
    # fig, ax1 = plt.subplots()
    # fig.canvas.set_window_title("Hidden Markov Model Graph")
    # hmm_supervised_pos.plot()
    # plt.show()
    #quit()
    # ###
    mapping = ['negneg','negneu','negpos','neuneg','neuneu','neupos','posneg','posneu','pospos','start','end']


    unseen_factor_smoothing = 0.1
    predicted = []
    test_data_size = len(test_data)
    count_newunseen = 0
    count_problematic = 0
    empty_sequences = 0

    #print(hmm_supervised_pos.states[1].distribution.parameters[0]["oh"])

    # MATH: score = π(state1) * EmmisionProb(o1|state1) * P(o2|o1) * EmmisionProb(o2|state2) * P(o3|o2) * ...  / SequenceLength

    # NO DUMMY STATES, SIMPLER CODE

    for k in range(test_data_size):
        current_observations = test_data[k]
        current_states = test_data[k]
        # Debug
        # print(k, len(current_observations))

        if len(current_observations) > 0:         
            sentiment_score_pos = transition_proba_matrix_pos[mapping.index("start"), mapping.index(current_states[0])]  # Transition from start to first state
            sentiment_score_neg = transition_proba_matrix_neg[mapping.index("start"), mapping.index(current_states[0])]  # Transition from start to first state
            sentiment_score_neu = transition_proba_matrix_neu[mapping.index("start"), mapping.index(current_states[0])]  # Transition from start to first state

            for i in range(len(current_observations)-1):

                current_state_ind = mapping.index(current_states[i])
                next_state_ind = mapping.index(current_states[i+1])

                try:        
                    emissionprob_pos = hmm_supervised_pos.states[current_state_ind].distribution.parameters[0][current_observations[i]]
                except KeyError as err:  # Prediction failed, we stumbled upon new unseen observation, set a manual probability
                    count_newunseen += 1        
                    print("Prediction Failed, new unseen observation:", err)
                    emissionprob_pos = unseen_factor_smoothing

                try:        
                    emissionprob_neg = hmm_supervised_neg.states[current_state_ind].distribution.parameters[0][current_observations[i]]
                except KeyError as err:  # Prediction failed, we stumbled upon new unseen observation, set a manual probability
                    print("Prediction Failed, new unseen observation:", err)
                    count_newunseen += 1  
                    emissionprob_neg = unseen_factor_smoothing

                try:        
                    emissionprob_neu = hmm_supervised_neu.states[current_state_ind].distribution.parameters[0][current_observations[i]]
                except KeyError as err:  # Prediction failed, we stumbled upon new unseen observation, set a manual probability
                    print("Prediction Failed, new unseen observation:", err)
                    count_newunseen += 1  
                    emissionprob_neu = unseen_factor_smoothing


                trans_prob_pos = transition_proba_matrix_pos[current_state_ind, next_state_ind]
                trans_prob_neg = transition_proba_matrix_neg[current_state_ind, next_state_ind]
                trans_prob_neu = transition_proba_matrix_neu[current_state_ind, next_state_ind]

                sentiment_score_pos = sentiment_score_pos * emissionprob_pos * trans_prob_pos
                sentiment_score_neg = sentiment_score_neg * emissionprob_neg * trans_prob_neg
                sentiment_score_neu = sentiment_score_neu * emissionprob_neu * trans_prob_neu                
            # last state with no transition          
            current_state_ind = mapping.index(current_states[-1])
            try:  
                sentiment_score_pos *= hmm_supervised_pos.states[current_state_ind].distribution.parameters[0][current_observations[-1]]
            except KeyError as err:  # Prediction failed, we stumbled upon new unseen observation, set a manual probability
                count_newunseen += 1  
                print("Prediction Failed, new unseen observation:", err)
                sentiment_score_pos *= unseen_factor_smoothing
            try:  
                sentiment_score_neg *= hmm_supervised_neg.states[current_state_ind].distribution.parameters[0][current_observations[-1]]
            except KeyError as err:  # Prediction failed, we stumbled upon new unseen observation, set a manual probability
                count_newunseen += 1  
                print("Prediction Failed, new unseen observation:", err)
                sentiment_score_neg *= unseen_factor_smoothing
            try:  
                sentiment_score_neu *= hmm_supervised_neu.states[current_state_ind].distribution.parameters[0][current_observations[-1]]
            except KeyError as err:  # Prediction failed, we stumbled upon new unseen observation, set a manual probability
                count_newunseen += 1  
                print("Prediction Failed, new unseen observation:", err)
                sentiment_score_neu *= unseen_factor_smoothing                

            max_winner = max(sentiment_score_pos, sentiment_score_neg, sentiment_score_neu)
            # Debug
            #print("Scores are:", (sentiment_score_pos, sentiment_score_neg, sentiment_score_neu), "for:", current_observations)
            # Comparison
            if sentiment_score_pos == max_winner:
                predicted.append("pos")
            elif sentiment_score_neg == max_winner:
                predicted.append("neg")
            elif sentiment_score_neu == max_winner:
                predicted.append("neu")
            else:
                print("NOT ENOUGH TRAINING DATA OR SOMETHING, performing random guessing")
                count_problematic += 1     
                rng = random.randint(0, 2)
                if rng == 0:
                    predicted.append("pos")
                elif rng == 1:
                    predicted.append("neg")
                else:
                    predicted.append("neu")
                
        
        else:  # Empty Sequence
            print("EMPTY SEQUENCE, performing random guessing")
            empty_sequences += 1
            rng = random.randint(0, 2)
            if rng == 0:
                predicted.append("pos")
            elif rng == 1:
                predicted.append("neg")
            else:
                predicted.append("neu")

    #seq_count2 = x_test.shape[0]
    #multivariate_3d_test_matrix = np.empty([seq_count2, 1, feature_count])  # Feature Count Remains the same when we vectorize the test set

    Print_Result_Metrics(labels_test, predicted, targetnames, silent_enable_2, time_counter, 0, "HMM "+str(n_order)+"th Order Supervised")

    print("New unseen observations:", count_newunseen, "Problematic Sequences:", count_problematic, "Empty Sequences:", empty_sequences)

    return None
    ###


def Print_Result_Metrics(labels_test, predicted, targetnames, silent_enable, time_counter, not_new_model, model_name):
    '''    Print Metrics after Training (Testing phase)    '''
    global cross_validation_best, cross_validation_best_ensemble, cross_validation_all, time_complexity_average

    # Time
    if not_new_model == 0:
        time_final = my_time.time()-time_counter
    else:
        time_final = time_counter

    # Metrics
    accuracy = metrics.accuracy_score(labels_test, predicted)
    other_metrics_to_print = metrics.classification_report(labels_test, predicted, target_names=targetnames, output_dict=False)
    other_metrics_as_dict = metrics.classification_report(labels_test, predicted, target_names=targetnames, output_dict=True)
    confusion_matrix = metrics.confusion_matrix(labels_test, predicted)

    if silent_enable == 0:
        print('\n- - - - - RESULT METRICS -', "%.2fsec" % time_final, model_name, '- - - - -')
        print('Exact Accuracy: ', accuracy)
        print(other_metrics_to_print)
        print(confusion_matrix)
        print()

    # Save to Global Variables
    if not_new_model == 0:  # Lack of this leads to an important Bug; If this flag is 1 we are storing the same model twice
        weighted_f1 = other_metrics_as_dict['weighted avg']['f1-score']
        cross_validation_all[model_name].append((accuracy, weighted_f1))  # Tuple
        time_complexity_average[model_name].append(time_final)
        
        if accuracy > cross_validation_best[0]:
            cross_validation_best[0] = accuracy
            cross_validation_best[1] = weighted_f1
            cross_validation_best[2] = model_name
            cross_validation_best[3] = labels_test
            cross_validation_best[4] = predicted 
            cross_validation_best[5] = time_final 
        if model_name.startswith("Ensemble") == True:
            if accuracy > cross_validation_best_ensemble[0]:
                cross_validation_best_ensemble[0] = accuracy
                cross_validation_best_ensemble[1] = weighted_f1
                cross_validation_best_ensemble[2] = model_name
                cross_validation_best_ensemble[3] = labels_test
                cross_validation_best_ensemble[4] = predicted  
                cross_validation_best_ensemble[5] = time_final           


def Print_Result_CrossVal_Final(k):
    '''    Print the best and average results across all cross validations    '''
    global cross_validation_best, cross_validation_best_ensemble, cross_validation_all

    print()
    print("- " * 18, "CONCLUSION across", str(k), "\b-fold Cross Validation", "- " * 18, "\n")
    if cross_validation_best[0] > 0.0:
        print("- " * 18, "BEST SINGLE MODEL", "- " * 18)
        Print_Result_Metrics(cross_validation_best[3], cross_validation_best[4], None, 0, cross_validation_best[5], 1, cross_validation_best[2])
        if cross_validation_best_ensemble[0] > 0.0:
            print("- " * 18, "BEST ENSEMBLE MODEL", "- " * 18)
            Print_Result_Metrics(cross_validation_best_ensemble[3], cross_validation_best_ensemble[4], None, 0, cross_validation_best_ensemble[5], 1, cross_validation_best_ensemble[2])

    print("- " * 18, "AVERAGES", "- " * 18)
    for model in cross_validation_all:
        avg = tuple(np.mean(cross_validation_all[model], axis=0))
        print(model, ": Accuracy is", "{:0.4f}".format(avg[0]), "F1-score is", "{:0.4f}".format(avg[1]))
        cross_validation_average[model] = avg  # Save the average on a global variable


def Plot_Results(k, dataset_name):
    '''    Plot the Metrics of all Hidden Markov Models in a Graph    '''
    global cross_validation_average

    print("Plotting AVERAGES of Cross Validation...")
    indices = np.arange(len(cross_validation_average))
    scores_acc = []
    scores_f1 = []
    model_names = []
    for model in cross_validation_average:
        scores_acc.append(cross_validation_average[model][0]) 
        scores_f1.append(cross_validation_average[model][1])
        model_names.append(model)
             
    fig, ax1 = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(left=0.18, top=0.92, bottom=0.08)
    fig.canvas.set_window_title(dataset_name + " - Averages across " + str(k) + "-fold Cross Validation")

    p1 = ax1.bar(indices + 0.35, scores_acc, align="center", width=0.35, label="Accuracy (%)", color="navy")    
    p2 = ax1.bar(indices, scores_f1, align="center", width=0.35, label="Accuracy (%)", color="cornflowerblue")

    ax1.set_title(dataset_name + " - Averages across " + str(k) + "-fold Cross Validation")
    ax1.set_ylim([0, 1])
    ax1.yaxis.set_major_locator(MaxNLocator(11))
    ax1.yaxis.grid(True, linestyle='--', which="major", color="grey", alpha=.25)
    ax1.set_ylabel("Performance")    
    ax1.legend((p1[0], p2[0]), ("Accuracy", "F1-score"))

    ax1.set_xticks(indices + 0.35 / 2)
    ax1.set_xticklabels(model_names)

    # Rotates labels and aligns them horizontally to left 
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")

    # Automatically adjust subplot parameters so that the the subplot fits in to the figure area
    fig.tight_layout()

    plt.show()
    print()


### START

np.set_printoptions(precision=10)  # Numpy Precision when Printing

df_dataset = Run_Preprocessing("Finegrained Sentiment Dataset")
#all_data = df_dataset.loc[:,"Sequences"]
#all_labels = df_dataset.loc[:,"Labels"]

print("\n--Dataset Info:\n", df_dataset.describe(include="all"), "\n\n", df_dataset.head(), "\n\n", df_dataset.loc[:,'Labels'].value_counts(), "\n--\n", sep="")

# Split using k-fold Cross Validation
set_fold = 5
k_fold = RepeatedStratifiedKFold(set_fold, n_repeats=1, random_state=22)

for k, (train_indexes, test_indexes) in enumerate(k_fold.split(df_dataset.loc[:,"Sequences"], df_dataset.loc[:,"Labels"])):  # Split must be done before every classifier because generated object gets exhausted (destroyed)
    print("\n--Current Cross Validation Fold:", k+1)

    train_data = df_dataset.reindex(train_indexes, copy=True, axis=0)
    pos_data_train = train_data.loc[train_data["Labels"] == "pos", ['Sequences']]
    neg_data_train = train_data.loc[train_data["Labels"] == "neg", ['Sequences']]
    neu_data_train = train_data.loc[train_data["Labels"] == "neu", ['Sequences']]    
    data_test = df_dataset.loc[:,"Sequences"].reindex(test_indexes, copy=True, axis=0)
    labels_test = df_dataset.loc[:,"Labels"].reindex(test_indexes, copy=True, axis=0)

    ### LET'S BUILD : HIGH-ORDER HIDDEN MARKOV MODEL
    # sentenceSentiments = list(set(x for l in data_train for x in l))  # get Unique Sentiments
    # print ("\n--Number of Observed States is", len(sentenceSentiments))

    documentSentiments = sorted(train_data.Labels.unique())            # get Unique Sentiments, everything is mapped against this List
    print ("--Number of Hidden States is", len(documentSentiments)*len(documentSentiments))

    # Note 1: We can increase the accuracy of high-orders but decrease the respective accuracy of low-oders, if we give only the end of the sequence for prediction -> (data_test_transformed[x][-2:], algorithm='viterbi') because way too many predictions/transtions fail
    # Note 2: Using Parallelism with n_jobs at -1 gives big speed boost but reduces accuracy   
    # Parameters: ..., ..., ..., ..., ..., targetnames, n_jobs, algorithm, graph_print_enable, silent_enable, silent_enable_2, n_order
    set_algorithm = 'map'
    #predicted_proba_1 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 0, 0, 1)

    #print(pos_data_train[0:10])

    predicted_proba_2 = HMM_NthOrder_Supervised(pos_data_train, neg_data_train, neu_data_train, data_test, labels_test, documentSentiments, None, 1, set_algorithm, 0, 0, 0, 2)
    # predicted_proba_3 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 3)
    # predicted_proba_4 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 4)
    # predicted_proba_5 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 5)
    # predicted_proba_6 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 6)
    # predicted_proba_7 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 7)
    # predicted_proba_8 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 8) 
    # predicted_proba_9 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 9)
    # predicted_proba_10 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 10) 
    # ###


    # ### Ensemble
    # if set_algorithm != 'viterbi':  # Ensemble does not work on the Viterbi algorithm. It works Maximum a posteriori and its Log probability Matrix of Predictions
    #     time_counter = my_time.time()  
    #     ensemble = 0.40*predicted_proba_1.iloc[:, 0:len(documentSentiments)].values + 0.30*predicted_proba_2.iloc[:, 0:len(documentSentiments)].values + 0.30*predicted_proba_3.iloc[:, 0:len(documentSentiments)].values  # Weights taken from C. Quan, F. Ren [2016]      
    #     predicted_indexes = np.argmax(ensemble, axis=1)
    #     predicted = list()
    #     for x in predicted_indexes:  # Convert Indexes to Strings
    #         predicted.append(documentSentiments[x])

    #     Print_Result_Metrics(labels_test.tolist(), predicted, None, 0, time_counter, 0, "Ensemble of 1+2+3 Orders") 
    # ###   

    #break  # Disable Cross Validation

if set_algorithm == 'viterbi':
    print("\nWarning: Ensemble will not be performed on Viterbi, select Maximum a posteriori instead...\n")

Print_Result_CrossVal_Final(set_fold)
#Plot_Results(set_fold, "Finegrained Sentiment Dataset")
#print(time_complexity_average)