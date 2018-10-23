''' 
Sentiment Analysis: Text Classification using Hidden Markov Models
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import log
from random import randint
import time as my_time  # Required to avoid some sort of conflict with pomegranate

from pomegranate import *
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from nltk import ngrams


cross_validation_best = [0.0, "", [], [], 0.0]  # [Score, Model Name, Actual Labels, Predicted, Time]
cross_validation_best_ensemble = [0.000, "", [], [], 0.0]  # [Score, Model Name, Actual Labels, Predicted, Time]

def Run_Preprocessing(dataset_name):
    '''    Dataset Dependant Preprocessing    '''

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

    # Remove empty instances from DataFrame, actually affects accuracy
    emptyCells = df_dataset.loc[df_dataset.Sequences.map(len) < 1].index.values
    df_dataset = df_dataset.drop(emptyCells, axis=0).reset_index(drop=True)  # Reset_Index to make the row numbers be consecutive again

    return df_dataset


def HMM_NthOrder_Unsupervised_and_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, targetnames, n_jobs, plot_enable, silent_enable, silent_enable_2, n):
    '''    Create 2 Hidden Markov Models of Nth Order, first is Unsupervised, second is Supervised    '''
    '''    Returns:     Predicted log probability Matrix                                              '''
    time_counter = my_time.time()

    # The sequences are stored as a List in the Dataframe, time to transform them to the correct form
    data_train_transformed = list() 
    data_test_transformed = list()       
    if n == 1:  # No need to do ngrams on 1st Order
        data_train_transformed = data_train.tolist()
        # Same
        data_test_transformed = data_test.tolist()
    else:
        for x in data_train:
            ngrams_temp = ngrams(x, n)
            temp = list()
            if len(x) >= n:
                for grams in ngrams_temp:
                    temp.append("".join(grams))

            data_train_transformed.append(temp)   
        # Same
        for x in data_test:
            ngrams_temp = ngrams(x, n)
            temp = list()
            if len(x) >= n:
                for grams in ngrams_temp:
                    temp.append("".join(grams))

            data_test_transformed.append(temp)   


    ### (Unsupervised) Train
    # hmm_leanfrominput = HiddenMarkovModel.from_samples(DiscreteDistribution, len(documentSentiments), X=data_train_transformed, n_jobs=n_jobs, verbose=False, name="Finegrained HMM")

    # # Find out which which State number corresponds to pos/neg/neu respectively
    # positiveState = list()
    # negativeState = list()
    # neutralState = list()
    # print()
    # for x in range(0, len(documentSentiments)):
    #     if silent_enable != 1:
    #         print("State", hmm_leanfrominput.states[x].name, hmm_leanfrominput.states[x].distribution.parameters)
    #     temp_dict = hmm_leanfrominput.states[x].distribution.parameters[0]
    #     positiveState.append(temp_dict["p" * n])
    #     negativeState.append(temp_dict["n" * n])
    #     neutralState.append(temp_dict["u" * n])
    # positiveState = positiveState.index(max(positiveState))
    # negativeState = negativeState.index(max(negativeState))
    # neutralState = neutralState.index(max(neutralState))
    # print("Pos Index is", positiveState, "Neg Index is", negativeState, "Neu Index is", neutralState)
    # ###

    # ### (Unsupervised) Predict
    # predicted = list()
    # for x in range(0, len(data_test_transformed)):
    #     try:
    #         predict = hmm_leanfrominput.predict(data_test_transformed[x], algorithm='viterbi')
    #     except ValueError as err:  # Prediction failed, predict randomly
    #         print("Prediction Failed:", err)
    #         predict = [randint(0, 2)]

    #     if predict[-1] == positiveState:  # I only care about the last Prediction
    #         predicted.append("pos")
    #     elif predict[-1] == negativeState:
    #         predicted.append("neg")
    #     else:
    #         predicted.append("neu")

    #     #predicted.append(hmm_leanfrominput.states[predict[-1]].name)

    # #Print_Result_Metrics(labels_test.tolist(), predicted, targetnames, silent_enable_2, time_counter, 0, "HMM "+str(n)+"th Order Supervised")
    # ###

    print()

    ### (Supervised) Train
    # In this case, find out which State corresponds to pos/neg/neu before training   
    labels_supervised = list()
    for i, x in enumerate(labels_train):
        getlength = len(data_train_transformed[i])
        if x == "pos":  # state0 is pos, state1 is neg, state2 is neu
            labels_supervised.append(["s0"] * getlength)
        elif x == "neg":      
            labels_supervised.append(["s1"] * getlength)    
        else:
            labels_supervised.append(["s2"] * getlength)  
    positiveState = 0
    negativeState = 1
    neutralState = 2              

    hmm_leanfrominput_supervised_2 = HiddenMarkovModel.from_samples(DiscreteDistribution, len(documentSentiments), X=data_train_transformed, labels=labels_supervised, state_names=["s0", "s1", "s2"], n_jobs=n_jobs, verbose=False, name="Finegrained HMM")

    if silent_enable != 1:
        for x in range(0, len(documentSentiments)):
            print("State", hmm_leanfrominput_supervised_2.states[x].name, hmm_leanfrominput_supervised_2.states[x].distribution.parameters)
    print("Pos Index is", positiveState, "Neg Index is", negativeState, "Neu Index is", neutralState)
    ###

    ### (Supervised) Predict
    predicted = list()
    for x in range(0, len(data_test_transformed)):
        try:        
            predict = hmm_leanfrominput_supervised_2.predict(data_test_transformed[x], algorithm='viterbi')
        except ValueError as err:  # Prediction failed, predict randomly
            print("Prediction Failed:", err)
            predict = [randint(0, 2)]           

        if predict[-1] == positiveState:  # I only care about the last Prediction
            predicted.append("pos")
        elif predict[-1] == negativeState:
            predicted.append("neg")
        else:
            predicted.append("neu")

        #predicted.append(hmm_leanfrominput_supervised_2.states[predict[-1]].name)

    Print_Result_Metrics(labels_test.tolist(), predicted, targetnames, silent_enable_2, time_counter, 0, "HMM "+str(n)+"th Order Supervised")
    ###

    ### Graph Plotting
    if plot_enable == 1:
        fig = plt.figure()
        fig.suptitle("Graph")

        ax = plt.subplot(121)
        ax.set_title("Unsupervised")
        #hmm_leanfrominput.plot() 
        ax = plt.subplot(122)
        ax.set_title("Supervised")
        hmm_leanfrominput_supervised_2.plot()
        plt.show()
    ###

    print()

    ### (Supervised) Log Probabilities for Ensembling
    predicted_proba = pd.DataFrame.from_dict({'Pos_Prob': None, 'Neg_Prob': None, 'Neu_Prob': None, 'PhraseId': data_test})
    for x in range(0, len(data_test_transformed)):
        if len(data_test_transformed[x]) > 0:
            try:      
                temp = hmm_leanfrominput_supervised_2.predict_log_proba(data_test_transformed[x])[-1]
            except ValueError as err:  # Prediction failed, predict equal probabilities
                print("Prediction Failed:", err)
                temp = [log(1.0 / len(documentSentiments))] * len(documentSentiments)  # log of base e                 
        else:  #  Prediction would return: Sequence is Impossible
            temp = [log(1.0 / len(documentSentiments))] * len(documentSentiments)  # log of base e
        predicted_proba.iloc[x, positiveState] = temp[positiveState]
        predicted_proba.iloc[x, negativeState] = temp[negativeState]
        predicted_proba.iloc[x, neutralState] = temp[neutralState]
    ###
    return predicted_proba


def Print_Result_Metrics(labels_test, predicted, targetnames, silent_enable, time_counter, time_flag, model_name):
    '''    Print Metrics after Training etc.    '''
    global cross_validation_best

    if time_flag == 0:
        time_final = my_time.time()-time_counter
    else:
        time_final = time_counter

    accuracy = metrics.accuracy_score(labels_test, predicted)
    if silent_enable == 0:
        if time_counter == 0.0:
            print('\n- - - - - RESULT METRICS -', model_name, '- - - - -')
        else:
            print('\n- - - - - RESULT METRICS -', "%.2fsec" % time_final, model_name, '- - - - -')
        print('Exact Accuracy: ', accuracy)
        print(metrics.classification_report(labels_test, predicted, target_names=targetnames))
        print(metrics.confusion_matrix(labels_test, predicted))
        print()

    if accuracy > cross_validation_best[0]:
        cross_validation_best[0] = accuracy
        cross_validation_best[1] = model_name
        cross_validation_best[2] = labels_test
        cross_validation_best[3] = predicted 
        cross_validation_best[4] = time_final 
    if model_name == "Ensemble":
        if accuracy > cross_validation_best_ensemble[0]:
            cross_validation_best_ensemble[0] = accuracy
            cross_validation_best_ensemble[1] = model_name
            cross_validation_best_ensemble[2] = labels_test
            cross_validation_best_ensemble[3] = predicted  
            cross_validation_best_ensemble[4] = time_final           


def Print_Result_Best(k):
    '''    Print Metrics only of the best result that occured    '''
    global cross_validation_best

    if cross_validation_best[0] > 0.0:
        print("\n" + "- " * 37, end = "")
        Print_Result_Metrics(cross_validation_best[2], cross_validation_best[3], None, 0, cross_validation_best[4], 1, cross_validation_best[1] + ", best of " + str(k+1) + " Cross Validations")
        Print_Result_Metrics(cross_validation_best_ensemble[2], cross_validation_best_ensemble[3], None, 0, cross_validation_best_ensemble[4], 1, cross_validation_best_ensemble[1] + ", best of " + str(k+1) + " Cross Validations")


### START

np.set_printoptions(precision=10)  # Numpy Precision when Printing

df_dataset = Run_Preprocessing("Finegrained")
all_data = df_dataset.loc[:,'Sequences']
all_labels = df_dataset.loc[:,'Labels']

print("\n--Dataset Info:\n", df_dataset.describe(include="all"), "\n\n", df_dataset.head(), "\n")

# Split using Cross Validation
k_fold = RepeatedStratifiedKFold(4, n_repeats=1, random_state=22)

for k, (train_indexes, test_indexes) in enumerate(k_fold.split(all_data, all_labels)):  # Spit must be done before every classifier because enumerate actually destroys the object:
    print("\n--Current Cross Validation Fold:", k)

    data_train = all_data.reindex(train_indexes, copy=True, axis=0)
    labels_train = all_labels.reindex(train_indexes, copy=True, axis=0)
    data_test = all_data.reindex(test_indexes, copy=True, axis=0)
    labels_test = all_labels.reindex(test_indexes, copy=True, axis=0)

    ### LET'S BUILD : High-Order Hidden Markov Model
    sentenceSentiments = set(x for l in data_train for x in l)  # get Unique Sentiments
    print ("\n--Number of Observed States is", len(sentenceSentiments))

    documentSentiments = set(labels_train.unique())  # get Unique Sentiments
    print ("--Number of Hidden States is", len(documentSentiments))

    # Parameters: targetnames, n_jobs, plot_enable, silent_enable, silent_enable_2, n      Running in Parallel with n_jobs at -1 gives big speed boost but reduces accuracy
    predicted_proba_1 = HMM_NthOrder_Unsupervised_and_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, 0, 1, 0, 1)
    predicted_proba_2 = HMM_NthOrder_Unsupervised_and_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, 0, 1, 0, 2)
    predicted_proba_3 = HMM_NthOrder_Unsupervised_and_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, 0, 1, 0, 3)
    ###


    ### Ensemble
    time_counter = my_time.time()    
    ensemble = 0.40*predicted_proba_1[['Pos_Prob', 'Neg_Prob', 'Neu_Prob']].values + 0.30*predicted_proba_2[['Pos_Prob', 'Neg_Prob', 'Neu_Prob']].values + 0.30*predicted_proba_3[['Pos_Prob', 'Neg_Prob', 'Neu_Prob']].values  # Weights taken from C. Quan, F. Ren [2015]
    predicted_indexes = np.round(np.argmax(ensemble, axis=1)).astype(int)
    predicted = list()
    for x in predicted_indexes:  # Convert Indexes to Strings
        if x == 0:    # positiveState
            predicted.append("pos")
        elif x == 1:  # negativeState
            predicted.append("neg")
        else:         # neutralState
            predicted.append("neu")

    Print_Result_Metrics(labels_test.tolist(), predicted, None, 0, time_counter, 0, "Ensemble") 
    ###   
    #break  # Disable Cross Validation

Print_Result_Best(k)