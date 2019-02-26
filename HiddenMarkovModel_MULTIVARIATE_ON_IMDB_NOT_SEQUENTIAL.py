''' 
Sentiment Analysis: Text Classification using Hidden Markov Models
'''

from sklearn.feature_extraction.text import TfidfVectorizer

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


cross_validation_best = [0.0, 0.0, "", [], [], 0.0]           # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
cross_validation_best_ensemble = [0.0, 0.0, "", [], [], 0.0]  # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
cross_validation_all = defaultdict(list)                      # {Name: (Accuracy, F1-score), (Accuracy, F1-score), ...}
cross_validation_average = defaultdict(list)                  # {Name: (Avg(Accuracy), Avg(F1-score)), ...}
time_complexity_average = defaultdict(list)                   # {Name: [Avg(Train+Test_Time)]


def Run_Preprocessing(dataset_name):
    '''    Dataset Dependant Preprocessing    '''

    # 1. Load the Dataset
    load = pd.read_csv('./Datasets/IMDb Large Movie Review Dataset as CSV/imdb_master.csv', names=['Type', 'Data', 'Labels'], usecols=[1,2,3], encoding='latin1')

    df_dataset = load[load.Labels != "unsup"]
    df_dataset = df_dataset[["Labels", "Data", "Type"]]

    print("--Processed", df_dataset.shape[0], "documents", "\n--Dataset Name:", dataset_name)

    # 2. Remove empty instances from DataFrame, actually affects accuracy
    emptyCells = df_dataset.loc[df_dataset.loc[:,'Data'].map(len) < 1].index.values
    df_dataset = df_dataset.drop(emptyCells, axis=0).reset_index(drop=True)  # Reset_Index to make the row numbers be consecutive again

    # 3. Balance the Dataset by Undersampling
    # mask = df_dataset.loc[:,'Labels'] == "No"
    # df_dataset_to_undersample = df_dataset[mask].sample(n=1718, random_state=22)
    # df_dataset = df_dataset[~mask]
    # df_dataset = pd.concat([df_dataset, df_dataset_to_undersample], ignore_index=True)
    # df_dataset = df_dataset.sample(frac=1, random_state=22).reset_index(drop=True)

    # 4. Shuffle the Datasets, it seems to be too perfectly ordered
    df_dataset = df_dataset.sample(frac=0.50).reset_index(drop=True)

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


    vectorizer = TfidfVectorizer(max_features = 200)
    x_train = vectorizer.fit_transform(data_train)
    x_test = vectorizer.transform(data_test)

    seq_count = x_train.shape[0]
    feature_count = x_train.shape[1]
    multivariate_3d_train_matrix = np.empty([seq_count, 1, feature_count])  # 1d: Sequence Count, 2d: Sequence Length, 3d: Feature Count // (n_samples, n_length, n_dimensions)
    # WE MUST ENSURE ITS A LIST OF LISTS
    #single_labels_to_use = np.array([[] for _ in range(seq_count)], dtype=object)
    single_labels_to_use = [[] for i in range(seq_count)]

    for i in range(seq_count):
        multivariate_3d_train_matrix[i, 0, :] = x_train[i, 0]
        single_labels_to_use[i].append(labels_train.tolist()[i])

    #print(multivariate_3d_train_matrix.dtype)
    multivariate_3d_train_matrix = multivariate_3d_train_matrix.astype("complex128")
    #quit()

    # Training
    hmm_leanfrominput_supervised = HiddenMarkovModel.from_samples(DirichletDistribution, n_components=2, X=multivariate_3d_train_matrix, labels=single_labels_to_use, n_jobs=-1, state_names=["pos", "neg"])
    # Note: Algorithm used is Baum-Welch

    print(hmm_leanfrominput_supervised)
    # Testing
    seq_count2 = x_test.shape[0]
    multivariate_3d_test_matrix = np.empty([seq_count2, 1, feature_count])  # Feature Count Remains the same when we vectorize the test set
    for j in range(seq_count2):
        multivariate_3d_test_matrix[j, 0, :] = x_test[j, 0]

    predicted = []
    for x in range(0, len(data_test)):
        #if len(x_test[x]) > 0:   
        #print(multivariate_3d_test_matrix[x])
        #multivariate_3d_test_matrix[x, 0, 1] = 0.01 
        #print(multivariate_3d_test_matrix[x])
        predict = hmm_leanfrominput_supervised.predict(multivariate_3d_test_matrix[x])
        print(predict)

        predicted.append(documentSentiments[predict[-1]])

    Print_Result_Metrics(labels_test.tolist(), predicted, targetnames, silent_enable_2, time_counter, 0, "HMM "+str(n_order)+"th Order Supervised")

    return None

    ### Print information about the Hidden Markov Model such as the probability matrix and the hidden states
    print()
    if silent_enable != 1:
        for x in range(0, len(documentSentiments)):
            print("State", hmm_leanfrominput_supervised.states[x].name, hmm_leanfrominput_supervised.states[x].distribution.parameters)
    print("Indexes:", tuple(zip(documentSentiments, state_names)))
    ###

    ### Plot the Hidden Markov Model Graph
    if graph_print_enable == 1:
        fig, ax1 = plt.subplots()
        fig.canvas.set_window_title("Hidden Markov Model Graph")
        ax1.set_title(str(n_order) + "-th Order")
        hmm_leanfrominput_supervised.plot()
        plt.show()
    ###

    ### Supervised Prediction (Testing) - Uses either Viterbi or Maximum a posteriori. Ensembling cannot be used with Viterbi predictions.
    predicted = list()

    if algorithm == 'viterbi':
        for x in range(0, len(data_test_transformed)):
            if len(data_test_transformed[x]) > 0:        
                try:        
                    predict = hmm_leanfrominput_supervised.predict(data_test_transformed[x], algorithm='viterbi')
                except ValueError as err:  # Prediction failed, predict randomly
                    print("Prediction Failed:", err)
                    predict = [randint(0, len(documentSentiments)-1)] 
            else:  #  Prediction would be stuck at Starting State
                predict = [randint(0, len(documentSentiments)-1)] 

            predicted.append(documentSentiments[predict[-1]])  # I only care about the last transition/prediction
            # predicted.append(hmm_leanfrominput_supervised.states[predict[-1]].name)

        Print_Result_Metrics(labels_test.tolist(), predicted, targetnames, silent_enable_2, time_counter, 0, "HMM "+str(n_order)+"th Order Supervised")

        return None      
    else:
        # Dataframe that will be used for the Ensemble
        predicted_proba = pd.DataFrame.from_dict({'Data': data_test})
        for i in range(0, len(documentSentiments)):
            predicted_proba.insert(loc=i, column=documentSentiments[i], value=np.nan)

        for x in range(0, len(data_test_transformed)):
            if len(data_test_transformed[x]) > 0:
                try:      
                    temp = hmm_leanfrominput_supervised.predict_log_proba(data_test_transformed[x])
                    predict = temp[-1] # I only care about the last transition/prediction
                except ValueError as err:  # Prediction failed, predict equal probabilities
                    print("Prediction Failed:", err)
                    predict = [log(1.0 / len(documentSentiments))] * len(documentSentiments)  # log of base e                
            else:  #  Prediction would be stuck at Starting State
                predict = [log(1.0 / len(documentSentiments))] * len(documentSentiments)  # log of base e

            for j in range (0, len(documentSentiments)):
                predicted_proba.iloc[x, j] = predict[j] 

        # Find argmax on the probability matrix
        # This entire process would be equivalent to simply calling predict(data_test_transformed[x], algorithm='map'), but predict two times would be more costly
        predicted_indexes = np.argmax(predicted_proba.iloc[:, 0:len(documentSentiments)].values, axis=1)
        predicted = list()
        for x in predicted_indexes:  # Convert Indexes to Strings
            predicted.append(documentSentiments[x])

        Print_Result_Metrics(labels_test.tolist(), predicted, targetnames, silent_enable_2, time_counter, 0, "HMM "+str(n_order)+"th Order Supervised")   

        return predicted_proba
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

df_dataset = Run_Preprocessing("IMDb Large Movie Review Dataset")

print("\n--Dataset Info:\n", df_dataset.describe(include="all"), "\n\n", df_dataset.head(), "\n\n", df_dataset.loc[:,'Labels'].value_counts(), "\n--\n", sep="")

data_train = df_dataset[df_dataset.Type == "train"].loc[:, "Data"]
labels_train = df_dataset[df_dataset.Type == "train"].loc[:, "Labels"]
data_test = df_dataset[df_dataset.Type == "test"].loc[:, "Data"]
labels_test = df_dataset[df_dataset.Type == "test"].loc[:, "Labels"]

### LET'S BUILD : HIGH-ORDER HIDDEN MARKOV MODEL
sentenceSentiments = list(set(x for l in data_train for x in l))  # get Unique Sentiments
print ("\n--Number of Observed States is", len(sentenceSentiments))

documentSentiments = list(set(labels_train.unique()))             # get Unique Sentiments, everything is mapped against this List
print ("--Number of Hidden States is", len(documentSentiments))

# Note 1: We can increase the accuracy of high-orders but decrease the respective accuracy of low-oders, if we give only the end of the sequence for prediction -> (data_test_transformed[x][-2:], algorithm='viterbi') because way too many predictions/transtions fail
# Note 2: Using Parallelism with n_jobs at -1 gives big speed boost but reduces accuracy   
# Parameters: ..., ..., ..., ..., ..., targetnames, n_jobs, algorithm, graph_print_enable, silent_enable, silent_enable_2, n_order
set_algorithm = 'map'
predicted_proba_1 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 1)


# # NO CROSS VALIDATION, IT'S ALREADY SPLIT
# # Split using k-fold Cross Validation
# set_fold = 5
# k_fold = RepeatedStratifiedKFold(set_fold, n_repeats=1, random_state=22)

# for k, (train_indexes, test_indexes) in enumerate(k_fold.split(all_data, all_labels)):  # Split must be done before every classifier because generated object gets exhausted (destroyed)
#     print("\n--Current Cross Validation Fold:", k+1)

#     data_train = all_data.reindex(train_indexes, copy=True, axis=0)
#     labels_train = all_labels.reindex(train_indexes, copy=True, axis=0)
#     data_test = all_data.reindex(test_indexes, copy=True, axis=0)
#     labels_test = all_labels.reindex(test_indexes, copy=True, axis=0)


#     ### LET'S BUILD : HIGH-ORDER HIDDEN MARKOV MODEL
#     sentenceSentiments = list(set(x for l in data_train for x in l))  # get Unique Sentiments
#     print ("\n--Number of Observed States is", len(sentenceSentiments))

#     documentSentiments = list(set(labels_train.unique()))             # get Unique Sentiments, everything is mapped against this List
#     print ("--Number of Hidden States is", len(documentSentiments))

#     # Note 1: We can increase the accuracy of high-orders but decrease the respective accuracy of low-oders, if we give only the end of the sequence for prediction -> (data_test_transformed[x][-2:], algorithm='viterbi') because way too many predictions/transtions fail
#     # Note 2: Using Parallelism with n_jobs at -1 gives big speed boost but reduces accuracy   
#     # Parameters: ..., ..., ..., ..., ..., targetnames, n_jobs, algorithm, graph_print_enable, silent_enable, silent_enable_2, n_order
#     set_algorithm = 'map'
#     predicted_proba_1 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 1)
#     predicted_proba_2 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 2)
#     predicted_proba_3 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 3)
#     predicted_proba_4 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 4)
#     predicted_proba_5 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 5)
#     predicted_proba_6 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 6)
#     predicted_proba_7 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 7)
#     predicted_proba_8 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 8) 
#     predicted_proba_9 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 9)
#     predicted_proba_10 = HMM_NthOrder_Supervised(data_train, data_test, labels_train, labels_test, documentSentiments, None, 1, set_algorithm, 0, 1, 0, 10) 
#     ###


#     ### Ensemble
#     if set_algorithm != 'viterbi':  # Ensemble does not work on the Viterbi algorithm. It works Maximum a posteriori and its Log probability Matrix of Predictions
#         time_counter = my_time.time()  
#         ensemble = 0.40*predicted_proba_1.iloc[:, 0:len(documentSentiments)].values + 0.30*predicted_proba_2.iloc[:, 0:len(documentSentiments)].values + 0.30*predicted_proba_3.iloc[:, 0:len(documentSentiments)].values  # Weights taken from C. Quan, F. Ren [2016]      
#         predicted_indexes = np.argmax(ensemble, axis=1)
#         predicted = list()
#         for x in predicted_indexes:  # Convert Indexes to Strings
#             predicted.append(documentSentiments[x])

#         Print_Result_Metrics(labels_test.tolist(), predicted, None, 0, time_counter, 0, "Ensemble of 1+2+3 Orders") 
#     ###   

#     #break  # Disable Cross Validation

# if set_algorithm == 'viterbi':
#     print("\nWarning: Ensemble will not be performed on Viterbi, select Maximum a posteriori instead...\n")

Print_Result_CrossVal_Final(1)
#Plot_Results(set_fold, "IMDb Large Movie Review Dataset")
#print(time_complexity_average)