''' 
Sentiment Analysis: Text Classification using Hidden Markov Models
'''

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import itertools
import numpy as np
import pandas as pd
from math import log
from random import randint
from collections import defaultdict
import time as my_time  # Required to avoid some sort of conflict with pomegranate

from pomegranate import *
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from nltk import ngrams


# Both F1 and Accuracy
cross_validation_best = [0.0, 0.0, "", [], [], 0.0]           # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
cross_validation_best_ensemble = [0.0, 0.0, "", [], [], 0.0]  # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
cross_validation_all = defaultdict(list)                      # {Name: [Accuracy, F1-score], [Accuracy, F1-score], ...]
cross_validation_average = defaultdict(list)                  # {Name: [Avg(Accuracy), Avg(F1-score])]

def Run_Preprocessing(dataset_name):
    '''    Dataset Dependant Preprocessing    '''

    # 1. Load Dataset
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
                    if temp[0] in ["neg", "neu", "pos", "mix"]:
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

    return df_dataset


def HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, targetnames, n_jobs, algorithm, graph_print_enable, silent_enable, silent_enable_2, n_order):
    '''    Create a Hidden Markov Model of n-th Order, trained in a Supervised manner    '''
    '''    Input:     
                    data_train, data_test, labels_train, labels_test: Data and their labels
                    documentSentiments: A list of Unique Sentiments which are the Hidden States
                    targetnames: Optional list of labels to display on the testing/evaluation phase
                    n_jobs: The number of jobs to run in parallel. Value of 1 means use 1 processor and -1 means use all processors
                    algorithm: Prediction Algorithm, can be either:
                                Viterbi ('viterbi') or Maximum a posteriori ('map')
                    graph_print_enable: Enables the plotting the graph of the Hidden Markov Model
                    silent_enable: Enable/disable printing the entire matrix of observation probabilities
                    silent_enable_2
    
    
    L123                            '''


    '''    Output:     Log probability Matrix of Predictions                            '''

    time_counter = my_time.time()

    # The sequences are stored as a List in the Dataframe, time to transform them to the correct form
    data_train_transformed = list() 
    data_test_transformed = list()       
    if n_order == 1:  # No need to do ngrams on 1st Order
        data_train_transformed = data_train.tolist()
        # Same
        data_test_transformed = data_test.tolist()
    # High-Order
    else:
        for x in data_train:
            ngrams_temp = ngrams(x, n_order)
            temp = list()
            if len(x) >= n_order:
                for grams in ngrams_temp:
                    temp.append("".join(grams))

            data_train_transformed.append(temp)   
        # Same
        for x in data_test:
            ngrams_temp = ngrams(x, n_order)
            temp = list()
            if len(x) >= n_order:
                for grams in ngrams_temp:
                    temp.append("".join(grams))

            data_test_transformed.append(temp)   


    ### Supervised Training
    # In this case we need to find out which State corresponds to each label (pos/neg/neu) before training  
    labels_supervised = list()
    for i, x in enumerate(labels_train):
        getlength = len(data_train_transformed[i])
        state_name = "s" + str(documentSentiments.index(x))
        labels_supervised.append([state_name] * getlength)

    state_names = list()
    for i in range(0, len(documentSentiments)):
        state_names.append("s" + str(i))

    hmm_leanfrominput_supervised = HiddenMarkovModel.from_samples(DiscreteDistribution, len(documentSentiments), X=data_train_transformed, labels=labels_supervised, state_names=state_names, n_jobs=n_jobs, verbose=False, name="Finegrained HMM")
    # Note: Algorithm used is Baum-Welch
    ###

    ### Print information about the Hidden Markov Model such as the probability matrix and the hidden states
    print()
    if silent_enable != 1:
        for x in range(0, len(documentSentiments)):
            print("State", hmm_leanfrominput_supervised.states[x].name, hmm_leanfrominput_supervised.states[x].distribution.parameters)
    print("Indexes:", tuple(zip(documentSentiments, state_names)))
    ###

    ### Supervised Prediction (Testing)
    predicted = list()
    for x in range(0, len(data_test_transformed)):
        if len(data_test_transformed[x]) > 0:        
            try:        
                predict = hmm_leanfrominput_supervised.predict(data_test_transformed[x], algorithm='viterbi')
            except ValueError as err:  # Prediction failed, predict randomly
                print("Prediction Failed:", err)
                predict = [randint(0, len(documentSentiments)-1)] 
        else:  #  Prediction would be stuck at Starting State
            predict = [randint(0, len(documentSentiments)-1)] 

        predicted.append(documentSentiments[predict[-1]])  # I only care about the last Prediction

        #predicted.append(hmm_leanfrominput_supervised.states[predict[-1]].name)

    Print_Result_Metrics(labels_test.tolist(), predicted, targetnames, silent_enable_2, time_counter, 0, "HMM "+str(n_order)+"th Order Supervised")
    ###

    ### Graph Plotting
    if graph_print_enable == 1:
        fig = plt.figure()
        fig.suptitle("Graph")

        ax = plt.subplot(121)
        ax.set_title("Unsupervised")
        #hmm_leanfrominput.plot() 
        ax = plt.subplot(122)
        ax.set_title("Supervised")
        hmm_leanfrominput_supervised.plot()
        plt.show()
    ###

    print()

    ### (Supervised) Log Probabilities for Ensembling
    predicted_proba = pd.DataFrame.from_dict({'Data': data_test})
    for i in range(0, len(documentSentiments)):
        predicted_proba.insert(loc=i, column=documentSentiments[i], value=np.nan)

    for x in range(0, len(data_test_transformed)):
        if len(data_test_transformed[x]) > 0:
            try:      
                temp = hmm_leanfrominput_supervised.predict_log_proba(data_test_transformed[x])[-1]
            except ValueError as err:  # Prediction failed, predict equal probabilities
                print("Prediction Failed:", err)
                temp = [log(1.0 / len(documentSentiments))] * len(documentSentiments)  # log of base e                 
        else:  #  Prediction would be stuck at Starting State
            temp = [log(1.0 / len(documentSentiments))] * len(documentSentiments)  # log of base e

        for j in range (0, len(documentSentiments)):
            predicted_proba.iloc[x, j] = temp[j] 
    ###
    return predicted_proba


def Print_Result_Metrics(labels_test, predicted, targetnames, silent_enable, time_counter, time_flag, model_name):
    '''    Print Metrics after Training (Testing phase)    '''
    global cross_validation_best, cross_validation_best_ensemble, cross_validation_all

    # Time
    if time_flag == 0:
        time_final = my_time.time()-time_counter
    else:
        time_final = time_counter

    # Metrics
    accuracy = metrics.accuracy_score(labels_test, predicted)
    other_metrics_to_print = metrics.classification_report(labels_test, predicted, target_names=targetnames, output_dict=False)
    other_metrics_as_dict = metrics.classification_report(labels_test, predicted, target_names=targetnames, output_dict=True)
    confusion_matrix = metrics.confusion_matrix(labels_test, predicted)

    if silent_enable == 0:
        if time_counter == 0.0:
            print('\n- - - - - RESULT METRICS -', model_name, '- - - - -')
        else:
            print('\n- - - - - RESULT METRICS -', "%.2fsec" % time_final, model_name, '- - - - -')
        print('Exact Accuracy: ', accuracy)
        print(other_metrics_to_print)
        print(confusion_matrix)
        print()

    # Save to Global Variables
    weighted_f1 = other_metrics_as_dict['weighted avg']['f1-score']
    cross_validation_all[model_name].append((accuracy, weighted_f1))  # Tuple
    
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
             
    # Reverse the items to appear in correct order
    scores_acc.reverse()
    scores_f1.reverse()
    model_names.reverse()

    fig, ax1 = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(left=0.18, top=0.92, bottom=0.08)
    fig.canvas.set_window_title(dataset_name + " - Averages across " + str(k) + "-fold Cross Validation")

    p1 = ax1.barh(indices + 0.35, scores_acc, align="center", height=0.35, label="Accuracy (%)", color="navy", tick_label=model_names)    
    p2 = ax1.barh(indices, scores_f1, align="center", height=0.35, label="Accuracy (%)", color="cornflowerblue", tick_label=model_names)

    ax1.set_title(dataset_name + " - Averages across " + str(k) + "-fold Cross Validation")
    ax1.set_xlim([0, 1])
    ax1.xaxis.set_major_locator(MaxNLocator(11))
    ax1.xaxis.grid(True, linestyle='--', which="major", color="grey", alpha=.25)
    ax1.legend((p1[0], p2[0]), ("Accuracy", "F1-score"))

    # Right-hand Y-axis
    indices_new = []
    for i in range(0, len(model_names)):  # Trick to print text on the y axis for both bars
        indices_new.append(indices[i])
        indices_new.append(indices[i] + 0.35) 

    ax2 = ax1.twinx()
    ax2.set_yticks(indices_new)
    ax2.set_ylim(ax1.get_ylim())  # Make sure that the limits are set equally on both yaxis so the ticks line up
    ax2.set_yticklabels(x for x in itertools.chain.from_iterable(itertools.zip_longest(scores_f1,scores_acc)) if x)  # Combine two lists in an alternating fashion
    ax2.set_ylabel("Performance")

    plt.show()
    print()


### START

np.set_printoptions(precision=10)  # Numpy Precision when Printing

df_dataset = Run_Preprocessing("Finegrained Sentiment Dataset")
all_data = df_dataset.loc[:,"Sequences"]
all_labels = df_dataset.loc[:,"Labels"]

print("\n--Dataset Info:\n", df_dataset.describe(include="all"), "\n\n", df_dataset.head(), "\n\n", df_dataset.loc[:,'Labels'].value_counts(), "\n--\n", sep="")

# Split using k-fold Cross Validation
set_fold = 5
k_fold = RepeatedStratifiedKFold(set_fold, n_repeats=1, random_state=22)

for k, (train_indexes, test_indexes) in enumerate(k_fold.split(all_data, all_labels)):  # Split must be done before every classifier because generated object gets exhausted (destroyed)
    print("\n--Current Cross Validation Fold:", k+1)

    data_train = all_data.reindex(train_indexes, copy=True, axis=0)
    labels_train = all_labels.reindex(train_indexes, copy=True, axis=0)
    data_test = all_data.reindex(test_indexes, copy=True, axis=0)
    labels_test = all_labels.reindex(test_indexes, copy=True, axis=0)


    ### LET'S BUILD : HIGH-ORDER HIDDEN MARKOV MODEL
    sentenceSentiments = list(set(x for l in data_train for x in l))  # get Unique Sentiments
    print ("\n--Number of Observed States is", len(sentenceSentiments))

    documentSentiments = list(set(labels_train.unique()))             # get Unique Sentiments, everything is mapped against this List
    print ("--Number of Hidden States is", len(documentSentiments))

    # Note 1:!!!!!!!!!!! CHECK VALIDITY OF THIS, IT SHOULD DECREASE - Note for very High Order: If way too many predictions fail, accuracy could increase (or even decrease) if only the end of the sequence is given      (data_test_transformed[x][-2:], algorithm='viterbi')
    # Note 2: Using Parallelism with n_jobs at -1 gives big speed boost but reduces accuracy   
    # Parameters: ..., ..., ..., ..., ..., targetnames, n_jobs, algorithm, graph_print_enable, silent_enable, silent_enable_2, n_order
    predicted_proba_1 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, "map", 0, 1, 0, 1)
    predicted_proba_2 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, "map", 0, 1, 0, 2)
    predicted_proba_3 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, "map", 0, 1, 0, 3)
    #predicted_proba_4 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, 0, 1, 0, 4)
    #predicted_proba_5 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, 0, 1, 0, 5)
    #predicted_proba_6 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, 0, 1, 0, 6)
    #predicted_proba_7 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, 0, 1, 0, 7)
    #predicted_proba_8 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, 0, 1, 0, 8)
    #predicted_proba_9 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, 0, 1, 0, 9)
    #predicted_proba_10 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, 0, 1, 0, 10)
    ###


    ### Ensemble
    time_counter = my_time.time()  
    ensemble = 0.40*predicted_proba_1.iloc[:, 0:len(documentSentiments)].values + 0.30*predicted_proba_2.iloc[:, 0:len(documentSentiments)].values + 0.30*predicted_proba_3.iloc[:, 0:len(documentSentiments)].values  # Weights taken from C. Quan, F. Ren [2015]      
    predicted_indexes = np.round(np.argmax(ensemble, axis=1)).astype(int)
    predicted = list()
    for x in predicted_indexes:  # Convert Indexes to Strings
        predicted.append(documentSentiments[x])

    Print_Result_Metrics(labels_test.tolist(), predicted, None, 0, time_counter, 0, "Ensemble of 1+2+3 orders") 
    ###   

    #break  # Disable Cross Validation

Print_Result_CrossVal_Final(set_fold)
Plot_Results(set_fold, "Finegrained Sentiment Dataset")