''' 
Sentiment Analysis: Text Classification using Hidden Markov Models
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import log
from random import randint

from pomegranate import *
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from nltk import ngrams


cross_validation_best = [0.0, "", [], [], 0.0]  # [Score, Model Name, Actual Labels, Predicted, Time]
cross_validation_best_ensemble = [0.000, "", [], [], 0.0]  # [Score, Model Name, Actual Labels, Predicted, Time]

def Run_Preprocessing(dataset_name):
    '''    Dataset Dependant Preprocessing    '''

    dataset = [["" for j in range(3)] for i in range(294)]
    count = 0
    with open('./Datasets/Finegrained/finegrained.txt', 'r') as file:
        for line in file:
            if len(line.split("_")) == 3:
                dataset[count][0] = line.split("_")[1]
            elif len(line.strip()) == 0:
                count += 1
            else:
                temp = [x.strip() for x in line.split("\t")]
                if len(temp[1]) > 1:
                    # n for negative - u for neutral - p for positive - m for mix --- nr is ignored
                    if temp[0].startswith("neg"):
                        dataset[count][1] += "n"
                    elif temp[0].startswith("neu"):
                        dataset[count][1] += "u"
                    elif temp[0].startswith("pos"):
                        dataset[count][1] += "p"
                    elif temp[0].startswith("mix"):
                        dataset[count][1] += "m"                    

                    dataset[count][2] += temp[1]


    print("--Processed", count+1, "documents", "\n--Dataset Name:", dataset_name)

    df_dataset = pd.DataFrame(data=dataset)

    # Remove empty instances from DataFrame
    emptyCells = df_dataset.loc[df_dataset.iloc[:,1] == ''].index.values
    df_dataset = df_dataset.drop(emptyCells, axis=0).reset_index(drop=True)  # Reset_Index to make the row numbers be consecutive again

    return df_dataset


def HMM_1stOrder_Unsupervised_and_Supervised(data_train, data_test, labels_train, labels_test, silent_enable, plot_enable):
    '''    Create 2 Hidden Markov Models of 1st Order, first is Unsupervised, second is Supervised    '''
    '''    Returns:     Predicted log probability Matrix                                              '''

    ### (Unsupervised) Train
    # The sequences are stored as a String in the Dataframe, time to transform them to the correct form
    data_train_asList = list()
    for x in data_train:
        data_train_asList.append(list(x))
    # Same
    data_test_asList = list()
    for x in data_test:
        data_test_asList.append(list(x))    

    # hmm_leanfrominput = HiddenMarkovModel.from_samples(DiscreteDistribution, len(documentSentiments), X=data_train_asList, n_jobs=-1, verbose=False, name="FineGrained HMM")

    # # Find out which which State number corresponds to pos/neg/neu respectively
    # positiveState = list()
    # negativeState = list()
    # neutralState = list()
    # print()
    # for x in range(0, len(documentSentiments)):
    #     if silent_enable != 1:
    #         print("State", hmm_leanfrominput.states[x].name, hmm_leanfrominput.states[x].distribution.parameters)
    #     temp_dict = hmm_leanfrominput.states[x].distribution.parameters[0]
    #     positiveState.append(temp_dict["p"])
    #     negativeState.append(temp_dict["n"])
    #     neutralState.append(temp_dict["u"])
    # positiveState = positiveState.index(max(positiveState))
    # negativeState = negativeState.index(max(negativeState))
    # neutralState = neutralState.index(max(neutralState))
    # print("Pos Index is", positiveState, "Neg Index is", negativeState, "Neu Index is", neutralState)
    # ###

    # ### (Unsupervised) Predict
    # predicted = list()
    # for x in range(0, len(data_test_asList)):
    #     predict = hmm_leanfrominput.predict(data_test_asList[x], algorithm='viterbi')

    #     if predict[-1] == positiveState:  # I only care about the last Prediction
    #         predicted.append("pos")
    #     elif predict[-1] == negativeState:
    #         predicted.append("neg")
    #     else:
    #         predicted.append("neu")

    #     #predicted.append(hmm_leanfrominput.states[predict[-1]].name)

    # #Print_Result_Metrics(labels_test.tolist(), predicted, None, "HMM 1st Order Unsupervised")
    # ###

    print()

    ### (Supervised) Train
    # In this case, find out which State corresponds to pos/neg/neu before training
    labels_supervised = list()
    for i, x in enumerate(labels_train):
        getlength = len(data_train_asList[i])
        if x == "pos":  # state0 is pos, state1 is neg, state2 is neu
            labels_supervised.append(["s0"] * getlength)
        elif x == "neg":      
            labels_supervised.append(["s1"] * getlength)    
        else:
            labels_supervised.append(["s2"] * getlength)  
    positiveState = 0
    negativeState = 1
    neutralState = 2

    hmm_leanfrominput_supervised = HiddenMarkovModel.from_samples(DiscreteDistribution, len(documentSentiments), X=data_train_asList, labels=labels_supervised, state_names=["s0", "s1", "s2"], n_jobs=-1, verbose=False, name="FineGrained HMM")

    if silent_enable != 1:
        for x in range(0, len(documentSentiments)):
            print("State", hmm_leanfrominput_supervised.states[x].name, hmm_leanfrominput_supervised.states[x].distribution.parameters)
    print("Pos Index is", positiveState, "Neg Index is", negativeState, "Neu Index is", neutralState)
    ###

    ### (Supervised) Predict
    predicted = list()
    for x in range(0, len(data_test_asList)):
        predict = hmm_leanfrominput_supervised.predict(data_test_asList[x], algorithm='viterbi')

        if predict[-1] == positiveState:  # I only care about the last Prediction
            predicted.append("pos")
        elif predict[-1] == negativeState:
            predicted.append("neg")
        else:
            predicted.append("neu")

        #predicted.append(hmm_leanfrominput_supervised.states[predict[-1]].name)

    Print_Result_Metrics(labels_test.tolist(), predicted, None, "HMM 1st Order Supervised")
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
        hmm_leanfrominput_supervised.plot()
        plt.show()
    ###

    print()

    ### (Supervised) Log Probabilities for Ensembling
    predicted_proba = pd.DataFrame.from_dict({'Pos_Prob': None, 'Neg_Prob': None, 'Neu_Prob': None, 'PhraseId': data_test})
    for x in range(0, len(data_test_asList)):
        if len(data_test_asList[x]) > 0:
            temp = hmm_leanfrominput_supervised.predict_log_proba(data_test_asList[x])[-1]
        else:  #  Prediction would return: Sequence is Impossible
            temp = [log(1.0 / len(documentSentiments))] * len(documentSentiments)  # log of base e
        predicted_proba.iloc[x, positiveState] = temp[positiveState]
        predicted_proba.iloc[x, negativeState] = temp[negativeState]
        predicted_proba.iloc[x, neutralState] = temp[neutralState]      
    ###
    return predicted_proba


def HMM_NthOrder_Unsupervised_and_Supervised(data_train, data_test, labels_train, labels_test, silent_enable, plot_enable, n):
    '''    Create 2 Hidden Markov Models of Nth Order, first is Unsupervised, second is Supervised    '''
    '''    Returns:     Predicted log probability Matrix                                              '''

    ### (Unsupervised) Train
    # The sequences are stored as a String in the Dataframe, time to transform them to the correct form
    data_train_ngram = list()
    for x in data_train:
        ngrams_temp = ngrams(x, n)
        temp = list()
        if len(x) >= n:
            for grams in ngrams_temp:
              temp.append("".join(grams))

        data_train_ngram.append(temp)   
    # Same
    data_test_ngram = list()
    for x in data_test:
        ngrams_temp = ngrams(x, n)
        temp = list()
        if len(x) >= n:
            for grams in ngrams_temp:
              temp.append("".join(grams))

        data_test_ngram.append(temp)   

    # hmm_leanfrominput_2 = HiddenMarkovModel.from_samples(DiscreteDistribution, len(documentSentiments), X=data_train_ngram, n_jobs=-1, verbose=False, name="FineGrained HMM")

    # # Find out which which State number corresponds to pos/neg/neu respectively
    # positiveState = list()
    # negativeState = list()
    # neutralState = list()
    # print()
    # for x in range(0, len(documentSentiments)):
    #     if silent_enable != 1:
    #         print("State", hmm_leanfrominput_2.states[x].name, hmm_leanfrominput_2.states[x].distribution.parameters)
    #     temp_dict = hmm_leanfrominput_2.states[x].distribution.parameters[0]
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
    # for x in range(0, len(data_test_ngram)):
    #     try:
    #         predict = hmm_leanfrominput_2.predict(data_test_ngram[x], algorithm='viterbi')
    #     except ValueError as err:  # Prediction failed, predict randomly
    #         print("Prediction Failed:", err)
    #         predict = [randint(0, 2)]

    #     if predict[-1] == positiveState:  # I only care about the last Prediction
    #         predicted.append("pos")
    #     elif predict[-1] == negativeState:
    #         predicted.append("neg")
    #     else:
    #         predicted.append("neu")

    #     #predicted.append(hmm_leanfrominput_2.states[predict[-1]].name)

    # #Print_Result_Metrics(labels_test.tolist(), predicted, None, "HMM "+str(n)+"th Order Unsupervised")
    # ###

    print()

    ### (Supervised) Train
    # In this case, find out which State corresponds to pos/neg/neu before training   
    labels_supervised = list()
    for i, x in enumerate(labels_train):
        getlength = len(data_train_ngram[i])
        if x == "pos":  # state0 is pos, state1 is neg, state2 is neu
            labels_supervised.append(["s0"] * getlength)
        elif x == "neg":      
            labels_supervised.append(["s1"] * getlength)    
        else:
            labels_supervised.append(["s2"] * getlength)  
    positiveState = 0
    negativeState = 1
    neutralState = 2              

    hmm_leanfrominput_supervised_2 = HiddenMarkovModel.from_samples(DiscreteDistribution, len(documentSentiments), X=data_train_ngram, labels=labels_supervised, state_names=["s0", "s1", "s2"], n_jobs=-1, verbose=False, name="FineGrained HMM")

    if silent_enable != 1:
        for x in range(0, len(documentSentiments)):
            print("State", hmm_leanfrominput_supervised_2.states[x].name, hmm_leanfrominput_supervised_2.states[x].distribution.parameters)
    print("Pos Index is", positiveState, "Neg Index is", negativeState, "Neu Index is", neutralState)
    ###

    ### (Supervised) Predict
    predicted = list()
    for x in range(0, len(data_test_ngram)):
        try:        
            predict = hmm_leanfrominput_supervised_2.predict(data_test_ngram[x], algorithm='viterbi')
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

    Print_Result_Metrics(labels_test.tolist(), predicted, None, "HMM "+str(n)+"th Order Supervised")
    ###

    ### Graph Plotting
    if plot_enable == 1:
        fig = plt.figure()
        fig.suptitle("Graph")

        ax = plt.subplot(121)
        ax.set_title("Unsupervised")
        #hmm_leanfrominput_2.plot() 
        ax = plt.subplot(122)
        ax.set_title("Supervised")
        hmm_leanfrominput_supervised_2.plot()
        plt.show()
    ###

    print()

    ### (Supervised) Log Probabilities for Ensembling
    predicted_proba = pd.DataFrame.from_dict({'Pos_Prob': None, 'Neg_Prob': None, 'Neu_Prob': None, 'PhraseId': data_test})
    for x in range(0, len(data_test_ngram)):
        if len(data_test_ngram[x]) > 0:
            try:      
                temp = hmm_leanfrominput_supervised_2.predict_log_proba(data_test_ngram[x])[-1]
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


def Print_Result_Metrics(labels_test, predicted, targetnames, model_name):
    '''    Print Metrics after Training etc.    '''
    global cross_validation_best

    print('\n- - - - - RESULT METRICS -', model_name, '- - - - -')
    accuracy = metrics.accuracy_score(labels_test, predicted)
    print('Exact Accuracy: ', accuracy)
    print(metrics.classification_report(labels_test, predicted, target_names=targetnames))
    print(metrics.confusion_matrix(labels_test, predicted))

    if accuracy > cross_validation_best[0]:
        cross_validation_best[0] = accuracy
        cross_validation_best[1] = model_name
        cross_validation_best[2] = labels_test
        cross_validation_best[3] = predicted 
    if model_name == "Ensemble":
        if accuracy > cross_validation_best_ensemble[0]:
            cross_validation_best_ensemble[0] = accuracy
            cross_validation_best_ensemble[1] = model_name
            cross_validation_best_ensemble[2] = labels_test
            cross_validation_best_ensemble[3] = predicted            


### START

df_dataset = Run_Preprocessing("Finegrained")
all_data = df_dataset.iloc[:,1]
all_labels = df_dataset.iloc[:,0]

print("\n--Dataset Info:\n", df_dataset.describe(include="all"), "\n\n", df_dataset.head(), "\n")

# Split using Cross Validation
k_fold = RepeatedStratifiedKFold(5, n_repeats=1, random_state=22)

for k, (train_indexes, test_indexes) in enumerate(k_fold.split(all_data, all_labels)):
    print("\n--Current Cross Validation Fold:", k)

    data_train = all_data.reindex(train_indexes, copy=True, axis=0)
    labels_train = all_labels.reindex(train_indexes, copy=True, axis=0)
    data_test = all_data.reindex(test_indexes, copy=True, axis=0)
    labels_test = all_labels.reindex(test_indexes, copy=True, axis=0)


    ### LET'S BUILD : High-Order Hidden Markov Model
    sentenceSentiments = set("".join(list(data_train.unique())))  # get Unique Sentiments
    print ("\n--Number of Observed States is", len(sentenceSentiments))

    documentSentiments = set(labels_train.unique())  # get Unique Sentiments
    print ("--Number of Hidden States is", len(documentSentiments))

    predicted_proba_1 = HMM_1stOrder_Unsupervised_and_Supervised(data_train, data_test, labels_train, labels_test, 1, 0)
    predicted_proba_2 = HMM_NthOrder_Unsupervised_and_Supervised(data_train, data_test, labels_train, labels_test, 1, 0, 2)
    predicted_proba_3 = HMM_NthOrder_Unsupervised_and_Supervised(data_train, data_test, labels_train, labels_test, 1, 0, 3)

    ### Ensemble
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

    Print_Result_Metrics(labels_test.tolist(), predicted, None, "Ensemble")
    ###


# Best Cross Validations
print("\n\n" + "- " * 37, end = "")
Print_Result_Metrics(cross_validation_best[2], cross_validation_best[3], None, "Best Overall: " + cross_validation_best[1])
Print_Result_Metrics(cross_validation_best_ensemble[2], cross_validation_best_ensemble[3], None, "Best Ensemble")
