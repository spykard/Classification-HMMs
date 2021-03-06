''' 
Sentiment Analysis: Text Classification using Hidden Markov Models
'''

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import ComplementNB
from sklearn.externals import joblib
import spacy
from nltk.tokenize import word_tokenize
import pickle
import random
import multiprocessing
import SimpleHOHMM
from SimpleHOHMM import HiddenMarkovModelBuilder as Builder

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import itertools
import numpy as np
import pandas as pd
from math import log
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


def Create_Artificial_Labels_to_File(data_train, labels_train, vocab, pipeline):
    # Training Data Transform
    pos_artifically_labeled_data = []
    pos_data_corresponding_to_labels = []
    neg_artifically_labeled_data = []
    neg_data_corresponding_to_labels = []
    golden_truth = []
    instance_count = len(data_train)
    nlp = spacy.load('en_core_web_sm')

    for i in range(instance_count):
        print("Currently in instance:", i, "of Train set")
        # CHANGE THIS TO BE OUTSIDE THE LOOP TO MAKE IT FASTER
        data_train_new = data_train.tolist()
        labels_train_new = labels_train.tolist()
        #print(data_train_new[0], "\n", data_train_new[1])
        tokenize_it = word_tokenize(data_train_new[i])
        to_append_labels = []
        to_append_data = []
        for j in tokenize_it:
            #print(i)
            token_to_string = str(j)
            if token_to_string in vocab:
                to_append_data.append(token_to_string)
                # we can simply directly append the artificial label itself
                prediction_bayes = pipeline.predict([token_to_string])[0]
                to_append_labels.append(prediction_bayes)
        
        if labels_train_new[i] == "pos":
            pos_artifically_labeled_data.append(to_append_labels)
            pos_data_corresponding_to_labels.append(to_append_data)
        elif labels_train_new[i] == "neg":
            neg_artifically_labeled_data.append(to_append_labels)
            neg_data_corresponding_to_labels.append(to_append_data)

        golden_truth.append(labels_train_new[i])

    # Pos
    with open('./Pickled Objects/To Train Pos HMM/Artificial_Labels_from_Bayes', 'wb') as f:
        pickle.dump(pos_artifically_labeled_data, f)
    with open('./Pickled Objects/To Train Pos HMM/Data_corresponding_to_Labels_from_Bayes', 'wb') as f:
        pickle.dump(pos_data_corresponding_to_labels, f)
    # Neg
    with open('./Pickled Objects/To Train Neg HMM/Artificial_Labels_from_Bayes', 'wb') as f:
        pickle.dump(neg_artifically_labeled_data, f)
    with open('./Pickled Objects/To Train Neg HMM/Data_corresponding_to_Labels_from_Bayes', 'wb') as f:
        pickle.dump(neg_data_corresponding_to_labels, f)
    #        
    #with open('./Pickled Objects/Artifical_Labels_Golden_Truth', 'wb') as f:
    #    pickle.dump(golden_truth, f)


def Create_Artificial_Labels_to_File_Test(data_test, labels_test, vocab, pipeline):
    # Test Data Transform
    artifically_labeled_data_test = []
    data_corresponding_to_labels_test = []
    golden_truth = []
    instance_count = len(data_test)
    nlp = spacy.load('en_core_web_sm')

    for i in range(instance_count):
        print("Currently in instance:", i, "of Test set")
        # some retarded bug
        data_test_new = data_test.tolist()
        labels_test_new = labels_test.tolist()
        #print(data_train_new[0], "\n", data_train_new[1])
        tokenize_it = word_tokenize(data_test_new[i])
        to_append_labels = []
        to_append_data = []
        for j in tokenize_it:
            #print(i)
            token_to_string = str(j)
            if token_to_string in vocab:
                to_append_data.append(token_to_string)
                # we can simply directly append the artificial label itself
                prediction_bayes = pipeline.predict([token_to_string])[0]
                to_append_labels.append(prediction_bayes)
        
        artifically_labeled_data_test.append(to_append_labels)
        data_corresponding_to_labels_test.append(to_append_data)
        golden_truth.append(labels_test_new[i])

    with open('./Pickled Objects/Artificial_Labels_from_Bayes_Test_Set', 'wb') as f:
        pickle.dump(artifically_labeled_data_test, f)
    with open('./Pickled Objects/Data_corresponding_to_Labels_from_Bayes_Test_Set', 'wb') as f:
        pickle.dump(data_corresponding_to_labels_test, f)
    with open('./Pickled Objects/Artifical_Labels_Golden_Truth_Test_Set', 'wb') as f:
        pickle.dump(golden_truth, f)


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


    pickle_load = 1    

    if pickle_load == 0:
        # STEP 1 RUN NAIVE BAYES
        pipeline = Pipeline([ # Optimal
                            ('vect', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode')),  # 1-Gram Vectorizer

                            ('tfidf', TfidfTransformer(use_idf=True)),
                            ('feature_selection', SelectKBest(score_func=chi2, k=2000)),  # Dimensionality Reduction                  
                            ('clf', ComplementNB()),])  
        
        pipeline.fit(data_train, labels_train)

        # (3) PREDICT
        predicted = pipeline.predict(data_test)

        accuracy = metrics.accuracy_score(labels_test, predicted)
        other_metrics_to_print = metrics.classification_report(labels_test, predicted, target_names=targetnames, output_dict=False)
        confusion_matrix = metrics.confusion_matrix(labels_test, predicted)
        print('Exact Accuracy: ', accuracy)
        print(other_metrics_to_print)
        print(confusion_matrix)

        vocab = pipeline.named_steps['vect'].get_feature_names()  # This is the total vocabulary
        selected_indices = pipeline.named_steps['feature_selection'].get_support(indices=True)  # This is the vocabulary after feature selection
        vocab = [vocab[i] for i in selected_indices]

        p1 = multiprocessing.Process(target=Create_Artificial_Labels_to_File, args=(data_train, labels_train, vocab, pipeline))
        p2 = multiprocessing.Process(target=Create_Artificial_Labels_to_File_Test, args=(data_test, labels_test, vocab, pipeline))
        p1.start()
        p2.start()
        quit()

    elif pickle_load == 1:

        pos_artifically_labeled_data = pickle.load(open('./Pickled Objects/To Train Pos HMM/Artificial_Labels_from_Bayes', 'rb'))
        pos_data_corresponding_to_labels = pickle.load(open('./Pickled Objects/To Train Pos HMM/Data_corresponding_to_Labels_from_Bayes', 'rb'))
        neg_artifically_labeled_data = pickle.load(open('./Pickled Objects/To Train Neg HMM/Artificial_Labels_from_Bayes', 'rb'))
        neg_data_corresponding_to_labels = pickle.load(open('./Pickled Objects/To Train Neg HMM/Data_corresponding_to_Labels_from_Bayes', 'rb'))

        artifically_labeled_data_test = pickle.load(open('./Pickled Objects/Artificial_Labels_from_Bayes_Test_Set', 'rb'))
        data_corresponding_to_labels_test = pickle.load(open('./Pickled Objects/Data_corresponding_to_Labels_from_Bayes_Test_Set', 'rb'))
        golden_truth_test = pickle.load(open('./Pickled Objects/Artifical_Labels_Golden_Truth_Test_Set', 'rb'))

    #print(len(data_corresponding_to_labels), len(data_corresponding_to_labels_test))

    # OPTION 1 WOULD BE TO USE TFIDF, 
    # OPTION 2 TO USE ONE-HOT ENCODING WITH DISCRETE
    # OPTION 3 WOULD BE TO USE NAIVE BAYES PREDICTION TO CREATE ARTIFICAL LABELS
    # WHATEVER

    # Code for Multivariate
    #multivariate_3d_train_matrix = np.empty([seq_count, 1, feature_count])  # 1d: Sequence Count, 2d: Sequence Length, 3d: Feature Count // (n_samples, n_length, n_dimensions)
    # WE MUST ENSURE ITS A LIST OF LISTS
    #single_labels_to_use = np.array([[] for _ in range(seq_count)], dtype=object)
    #single_labels_to_use = [[] for i in range(seq_count)]
    # for i in range(seq_count):
    #     multivariate_3d_train_matrix[i, 0, :] = x_train[i, 0]
    #     single_labels_to_use[i].append(labels_train.tolist()[i])
    #print(multivariate_3d_train_matrix.dtype)
    #multivariate_3d_train_matrix = multivariate_3d_train_matrix.astype("complex128")
    #quit()


    # LET'S BUILD: High-order HMM by converting representation to 1st order
    # creates dummy states. For a third-order model, we would need to add two dummy start states in high-order space
    # the prior probabilities of every state pair are set to zero except for more info see -> MINIHMM represent.py

    # second-order HMM
    # for second-order we add 1 dummy state
    enable_highorder = 1
    enable_pickle_load = 0

    if enable_highorder == 1 and enable_pickle_load == 0:
        set_order = 3

        # # Pos    
        # second_order_labels = []    
        # for seq in pos_artifically_labeled_data:
        #     tempp = []

        #     if len(seq) > 0:  # else remains empty
        #         # first            
        #         tempp.append("dummy" + seq[0])
        #         for i in range(set_order-1, len(seq)):
        #             # concatenate the names of 2 states
        #             tempp.append(seq[i-1] + seq[i])

        #     second_order_labels.append(tempp)
        # pos_artifically_labeled_data = second_order_labels  # Convert the name

        # # Neg    
        # second_order_labels = []    
        # for seq in neg_artifically_labeled_data:
        #     tempp = []

        #     if len(seq) > 0:  # else remains empty
        #         # first            
        #         tempp.append("dummy" + seq[0])
        #         for i in range(set_order-1, len(seq)):
        #             # concatenate the names of 2 states
        #             tempp.append(seq[i-1] + seq[i])

        #     second_order_labels.append(tempp)
        # neg_artifically_labeled_data = second_order_labels  # Convert the name

        # Test    
        # # THIS DOESN'T HAVE TO BE PERFECT 1 TO 1 MAPPING SINCE WE JUST MULTIPLY ANYWAY 
        second_order_labels = []    
        for seq in artifically_labeled_data_test:
            tempp = []

            if len(seq) > 0:  # else remains empty
                # first  
                tempp.append(seq[0]) 
                # PERFORMED CHANGE COMPARED TO SECOND-ORDER  
                if len(seq) > 1: 
                    tempp.append(seq[0] + "-" + seq[1]) 
                    for i in range(set_order-1, len(seq)):
                        # concatenate the names of 2 states
                        tempp.append(seq[i-2] + "-" + seq[i-1] + "-" + seq[i])

            second_order_labels.append(tempp)
        artifically_labeled_data_test = second_order_labels  # Convert the name

        # HIGH-ORDER
        # print(len(pos_artifically_labeled_data), len(neg_artifically_labeled_data))
        # print(pos_artifically_labeled_data[1])
        # print(pos_artifically_labeled_data[-1])
        # print("NOW NEG")
        # print(neg_artifically_labeled_data[1])
        # print(neg_artifically_labeled_data[-1])

        # Build Pos Class HMM - !!! state_names should be in alphabetical order
        #hmm_supervised_pos = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=6, X=pos_data_corresponding_to_labels, labels=pos_artifically_labeled_data, n_jobs=1, verbose=True)
        #print("NEXT HMM")
        # Build Neg Class HMM - !!! state_names should be in alphabetical order
        #hmm_supervised_neg = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=6, X=neg_data_corresponding_to_labels, labels=neg_artifically_labeled_data, n_jobs=2, verbose=True, max_iterations = 100)
    
        # with open('./Pickled Objects/High_Order_HMM_POS', 'wb') as f:
        #     pickle.dump(hmm_supervised_pos, f)
        # with open('./Pickled Objects/High_Order_HMM_NEG', 'wb') as f:
        #     pickle.dump(hmm_supervised_neg, f)
        # with open('./Pickled Objects/To Test HOHMM/High_Order_Test_Set', 'wb') as f:
        #     pickle.dump(artifically_labeled_data_test, f)

    elif enable_highorder == 1 and enable_pickle_load == 1:
        # hmm_supervised_pos = pickle.load(open('./Pickled Objects/High_Order_HMM_POS', 'rb'))
        # hmm_supervised_neg = pickle.load(open('./Pickled Objects/High_Order_HMM_NEG', 'rb'))   
        artifically_labeled_data_test = pickle.load(open('./Pickled Objects/To Test HOHMM/High_Order_Test_Set', 'rb')) 

    else:
        # Training
        # Build Pos Class HMM - !!! state_names should be in alphabetical order
        hmm_supervised_pos = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=2, X=pos_data_corresponding_to_labels, labels=pos_artifically_labeled_data, emission_pseudocount=0, n_jobs=1, state_names=["neg", "pos"])
        # Build Neg Class HMM - !!! state_names should be in alphabetical order
        hmm_supervised_neg = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=2, X=neg_data_corresponding_to_labels, labels=neg_artifically_labeled_data, emission_pseudocount=0, n_jobs=1, state_names=["neg", "pos"])
        # Note: Algorithm used is Baum-Welch 

    #print(neg_artifically_labeled_data[0])

    # Testing
    #print(len(pos_artifically_labeled_data), len(neg_artifically_labeled_data))
    #print(hmm_supervised_pos)
    #test_observ = ["realise", "pure"]
    #test_states = ["neg", "neg"]

    #transition_proba_matrix_pos = hmm_supervised_pos.dense_transition_matrix()
    #transition_proba_matrix_neg = hmm_supervised_neg.dense_transition_matrix()

    # Compare proba matrix to HOHMM

    set_pickle_load_2 = 1
    unseen_factor_smoothing = 0.5e-05  # Probability if we stumble upon new unseen observation     

    if set_pickle_load_2 == 0:

        # Try HOHMM
        builder = Builder()
        builder.add_batch_training_examples(pos_data_corresponding_to_labels, pos_artifically_labeled_data)
        # The smoothing factor number needs to be very specific and is found by looking at the values themselves e.g. through pomegranate
        hmm_pos = builder.build(highest_order = 3, k_smoothing=0.5e-05)  # or 9.0e-06
        builder.add_batch_training_examples(neg_data_corresponding_to_labels, neg_artifically_labeled_data)
        hmm_neg = builder.build(highest_order = 3, k_smoothing=0.5e-05)  # or 9.0e-06

        with open('./Pickled Objects/Third_Order_HOHMM_HMM_POS', 'wb') as f:
            pickle.dump(hmm_pos, f)
        with open('./Pickled Objects/Third_Order_HOHMM_HMM_NEG', 'wb') as f:
            pickle.dump(hmm_neg, f)           

    else:
        hmm_pos = pickle.load(open('./Pickled Objects/Third_Order_HOHMM_HMM_POS', 'rb'))
        hmm_neg = pickle.load(open('./Pickled Objects/Third_Order_HOHMM_HMM_NEG', 'rb'))
    # # Mapping the way he does
    # observation_mapping = set()
    # for obs_lst in pos_data_corresponding_to_labels:
    #     observation_mapping.update(set(obs_lst))

    # # State mapping is whichever he sees first

    #random.seed(22)

    hmm_p = hmm_pos.get_parameters()

    observation_mapping_pos = list(hmm_p["all_obs"])
    state_mapping_pos = list(hmm_p["all_states"])

    hmm_n = hmm_neg.get_parameters()

    observation_mapping_neg = list(hmm_n["all_obs"])
    state_mapping_neg = list(hmm_n["all_states"])

    # observation_mapping_neg = list(hmm_neg._all_obs)
    # state_mapping_neg = list(hmm_neg._all_states)
  
    # pi always includes all previous orders
    # B always hard stuck on [features x 2] refering only to the 2 states regardless of order

    #print(hmm_p["B"])

    if False:
        # Debugging
        print(hmm_p["pi"])
        print(hmm_p["A"])
        print(hmm_p["B"])

        print(len(hmm_p["B"]))
        print(len(hmm_p["B"][0]), len(hmm_p["B"][1]))
        quit()

 
    # !!! Debug the matrix to find which one is which
    # print(transition_proba_matrix_neg)
    # fig, ax1 = plt.subplots()
    # hmm_supervised_neg.plot()
    # plt.show() 
    # quit()
    # # Result :  
    # neg  [0]
    # pos  [1]
    # none-start [2]
    # none-end  [3]
    #data_corresponding_to_labels_test = data_corresponding_to_labels_test[0:500]
    #artifically_labeled_data_test = artifically_labeled_data_test[0:500]

    # PLOT HOW THIS VARIABLE AFFECTS PERFORMANCE
    predicted = []
    test_data_size = len(data_corresponding_to_labels_test)
    count_newunseen = 0
    count_problematic = 0
    empty_sequences = 0

    #print(hmm_supervised_pos.states[1].distribution.parameters[0]["oh"])

    # MATH: score = ??(state1) * EmmisionProb(o1|state1) * P(o2|o1) * EmmisionProb(o2|state2) * P(o3|o2) * ...  / SequenceLength

    for k in range(test_data_size):
        current_observations = data_corresponding_to_labels_test[k]
        # SKIP the ones we apply "pi" on
        current_states = artifically_labeled_data_test[k][2:]
        # Debug
        # print(k, len(current_observations))

        # CHANGE PERFORMED COMPARED TO SINGLE-ORDER ON FOLLOWING LINE
        if len(current_observations) > 0 and len(current_states) > 1:     
            # CHANGE PERFORMED COMPARED TO SINGLE-ORDER ON FOLLOWING LINE  
            sentiment_score_pos = hmm_p["pi"][0][artifically_labeled_data_test[k][0]]  # Transition from start to first state
            sentiment_score_neg = hmm_n["pi"][0][artifically_labeled_data_test[k][0]]  # Transition from start to first state
            # CHANGE PERFORMED COMPARED TO SECOND-ORDER ON FOLLOWING LINE 
            # now we utilize the pi twice
            sentiment_score_pos = hmm_p["pi"][1][artifically_labeled_data_test[k][1]]  # Transition from start to first state
            sentiment_score_neg = hmm_n["pi"][1][artifically_labeled_data_test[k][1]]  # Transition from start to first state            


            for i in range(len(current_observations) - 3):  # !!! CHANGED TO -3 FROM -2 might be missing a single emission because of this
                #print(i, len(current_states))
                #print(current_states)
                #print(current_states[i-3:i+3])
                #print(len(current_observations), len(current_states))

                current_state_ind_pos = state_mapping_pos.index(current_states[i])
                next_state_ind_pos = state_mapping_pos.index(current_states[i+1])

                current_state_ind_neg = state_mapping_neg.index(current_states[i])
                next_state_ind_neg = state_mapping_neg.index(current_states[i+1])                

                # CHANGE PERFORMED - I CHOOSE THE "pos" out of "pos-neg-neg" on every iteration
                manual_map = ["pos", "neg"]
                pick_first = manual_map.index(current_states[i][0:3])

                try:
                    emission_temp_ind = observation_mapping_pos.index(current_observations[i])       
                    emissionprob_pos = hmm_p["B"][pick_first][emission_temp_ind]
                except ValueError as err:  # Prediction failed, we stumbled upon new unseen observation, set a manual probability
                    count_newunseen += 1        
                    #print("Prediction Failed, new unseen observation:", err)
                    emissionprob_pos = unseen_factor_smoothing

                try:        
                    emission_temp_ind = observation_mapping_neg.index(current_observations[i])       
                    emissionprob_neg = hmm_n["B"][pick_first][emission_temp_ind]
                except ValueError as err:  # Prediction failed, we stumbled upon new unseen observation, set a manual probability
                    #print("Prediction Failed, new unseen observation:", err)
                    count_newunseen += 1  
                    emissionprob_neg = unseen_factor_smoothing

                trans_prob_pos = hmm_p["A"][current_state_ind_pos][next_state_ind_pos]
                trans_prob_neg = hmm_n["A"][current_state_ind_neg][next_state_ind_neg]

                sentiment_score_pos = sentiment_score_pos * emissionprob_pos * trans_prob_pos
                sentiment_score_neg = sentiment_score_neg * emissionprob_neg * trans_prob_neg

            # CHANGE PERFORMED ON FOLLOWING LINE    
            # last 3 states with no transition          

            for last3 in [-3, -2, -1]:

                if len(current_states) < 3 and last3 == -3:  # for too short sentences
                    continue     

                # CHANGE PERFORMED - I CHOOSE THE "pos" out of "pos-neg" on every iteration

                pick_first = manual_map.index(current_states[last3][0:3])
                try:  
                    emission_temp_ind = observation_mapping_pos.index(current_observations[last3])
                    sentiment_score_pos *= hmm_p["B"][pick_first][emission_temp_ind]
                except ValueError as err:  # Prediction failed, we stumbled upon new unseen observation, set a manual probability
                    count_newunseen += 1  
                    #print("Prediction Failed, new unseen observation:", err)
                    sentiment_score_pos *= unseen_factor_smoothing
                try:  
                    emission_temp_ind = observation_mapping_neg.index(current_observations[last3])  
                    sentiment_score_neg *= hmm_n["B"][pick_first][emission_temp_ind]
                except ValueError as err:  # Prediction failed, we stumbled upon new unseen observation, set a manual probability
                    count_newunseen += 1  
                    #print("Prediction Failed, new unseen observation:", err)
                    sentiment_score_neg *= unseen_factor_smoothing


            # Comparison
            if sentiment_score_pos > sentiment_score_neg:
                predicted.append("pos")
            elif sentiment_score_pos < sentiment_score_neg:
                predicted.append("neg")
            else:
                # print("NOT ENOUGH TRAINING DATA OR SOMETHING, performing random guessing")
                count_problematic += 1     
                rng = random.randint(0, 1)
                if rng == 0:
                    predicted.append("pos")
                else:
                    predicted.append("neg")
        
        else:  # Empty Sequence
            #print("EMPTY SEQUENCE, performing random guessing", artifically_labeled_data_test[k])
            empty_sequences += 1
            rng = random.randint(0, 1)
            if rng == 0:
                predicted.append("pos")
            else:
                predicted.append("neg")

    #seq_count2 = x_test.shape[0]
    #multivariate_3d_test_matrix = np.empty([seq_count2, 1, feature_count])  # Feature Count Remains the same when we vectorize the test set

    Print_Result_Metrics(golden_truth_test, predicted, targetnames, silent_enable_2, time_counter, 0, "HMM "+str(n_order)+"th Order Supervised")

    print("New unseen observations:", count_newunseen, "Problematic Sequences:", count_problematic, "Empty Sequences:", empty_sequences)

    # ### Print information about the Hidden Markov Model such as the probability matrix and the hidden states
    # print()
    # if silent_enable != 1:
    #     for x in range(0, len(documentSentiments)):
    #         print("State", hmm_leanfrominput_supervised.states[x].name, hmm_leanfrominput_supervised.states[x].distribution.parameters)
    # print("Indexes:", tuple(zip(documentSentiments, state_names)))
    # ###

    # ### Plot the Hidden Markov Model Graph
    # if graph_print_enable == 1:
    #     fig, ax1 = plt.subplots()
    #     fig.canvas.set_window_title("Hidden Markov Model Graph")
    #     ax1.set_title(str(n_order) + "-th Order")
    #     hmm_leanfrominput_supervised.plot()
    #     plt.show()
    # ### 

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