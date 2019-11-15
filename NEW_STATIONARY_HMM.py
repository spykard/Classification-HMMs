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
from random import randint
import multiprocessing
import SimpleHOHMM
from SimpleHOHMM import HiddenMarkovModelBuilder as Builder

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
    df_dataset = df_dataset.sample(frac=0.01, random_state=22).reset_index(drop=True)

    return df_dataset


def Create_Artificial_Labels_to_File(data_train, labels_train, vocab, pipeline):
    # Training Data Transform
    all_artifically_labeled_data = []
    all_data_corresponding_to_labels = []
    golden_truth = []
    instance_count = len(data_train)
    nlp = spacy.load('en_core_web_sm')

    # Get Sentiment Words from a generic Opinion Lexicon
    sentiment_words = []
    pos_words = []
    neg_words = []
    for line in open('./opinion_lexicon/positive-words.txt', 'r'):
        pos_words.append(line.rstrip())  # Must strip Newlines

    for line in open('./opinion_lexicon/negative-words.txt', 'r'):
        neg_words.append(line.rstrip())  # Must strip Newlines  

    sentiment_words = pos_words + neg_words
    sentiment_words = set(sentiment_words)

    # CHANGED THIS TO BE OUTSIDE THE LOOP 
    # (some retarded bug)
    data_train_new = data_train.tolist()
    labels_train_new = labels_train.tolist()

    for i in range(instance_count):
        print("Currently in instance:", i, "of Train set")

        #print(data_train_new[0], "\n", data_train_new[1])
        tokenize_it = word_tokenize(data_train_new[i])
        to_append_labels = []
        to_append_data = []
        for j in tokenize_it:
            #print(i)
            token_to_string = str(j)
            if token_to_string in sentiment_words:
                if token_to_string in vocab: ## Important to also be in vocabulary
                    to_append_data.append(token_to_string)
                    # we can simply directly append the artificial label itself
                    prediction_bayes = str(pipeline.predict([token_to_string])[0])  ## Convert from numpy.str_ to str
                    to_append_labels.append(prediction_bayes)
        
        all_artifically_labeled_data.append(to_append_labels)
        all_data_corresponding_to_labels.append(to_append_data)

        golden_truth.append(labels_train_new[i])

    # Save
    with open('./Pickled Objects/Artificial_Labels_from_Bayes', 'wb') as f:
        pickle.dump(all_artifically_labeled_data, f)
    with open('./Pickled Objects/Data_corresponding_to_Labels_from_Bayes', 'wb') as f:
        pickle.dump(all_data_corresponding_to_labels, f)       
    with open('./Pickled Objects/Artifical_Labels_Golden_Truth', 'wb') as f:
        pickle.dump(golden_truth, f)


def Create_Artificial_Labels_to_File_Test(data_test, labels_test, vocab, pipeline):
    # Test Data Transform
    artifically_labeled_data_test = []
    data_corresponding_to_labels_test = []
    golden_truth = []
    instance_count = len(data_test)
    nlp = spacy.load('en_core_web_sm')

    # Get Sentiment Words from a generic Opinion Lexicon
    sentiment_words = []
    pos_words = []
    neg_words = []
    for line in open('./opinion_lexicon/positive-words.txt', 'r'):
        pos_words.append(line.rstrip())  # Must strip Newlines

    for line in open('./opinion_lexicon/negative-words.txt', 'r'):
        neg_words.append(line.rstrip())  # Must strip Newlines  

    sentiment_words = pos_words + neg_words
    sentiment_words = set(sentiment_words)

    # CHANGED THIS TO BE OUTSIDE THE LOOP 
    # (some retarded bug)
    data_test_new = data_test.tolist()
    labels_test_new = labels_test.tolist()

    for i in range(instance_count):
        print("Currently in instance:", i, "of Test set")

        #print(data_train_new[0], "\n", data_train_new[1])
        tokenize_it = word_tokenize(data_test_new[i])
        to_append_labels = []
        to_append_data = []
        for j in tokenize_it:
            #print(i)
            token_to_string = str(j)
            if token_to_string in sentiment_words:            
                if token_to_string in vocab: ## Important to also be in vocabulary
                    to_append_data.append(token_to_string)
                    # we can simply directly append the artificial label itself
                    prediction_bayes = str(pipeline.predict([token_to_string])[0])  ## Convert from numpy.str_ to str
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


    pickle_load = 0    

    if pickle_load == 0:

        # Make it 80-20
        data = data_train.append(data_test)
        labels = labels_train.append(labels_test)
        
        naive_bayes_data_train, naive_bayes_data_test = np.split(data, [int(.8*len(data))])
        naive_bayes_labels_train, naive_bayes_labels_test = np.split(labels, [int(.8*len(labels))])

        # Stationary HMM on Entire Document first
        # print(naive_bayes_data_train[naive_bayes_labels_train == "pos"])

        # Let's get the features we are gonna use for all TF-IDFs
       
        vectorizer_general = TfidfVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode', use_idf=True)
        feature_selection = SelectKBest(score_func=chi2, k=500)
        temp_matrix = vectorizer_general.fit_transform(data)  # Entire data
        temp_matrix_2 = feature_selection.fit_transform(temp_matrix, labels)  # Entire labels

        vocab = vectorizer_general.get_feature_names()  # This is the total vocabulary
        selected_indices = feature_selection.get_support(indices=True)  # This is the vocabulary after feature selection
        vocab = [vocab[i] for i in selected_indices]


        # IDF train
        vectorizer_general_2 = TfidfVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode', use_idf=True, vocabulary=vocab)
        temp_matrix = vectorizer_general_2.fit_transform(naive_bayes_data_train)
        idf_train = vectorizer_general_2.idf_

        # IDF test
        # same as above

        # TF(polarity_1)
        vectorizer_pos = TfidfVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode', use_idf=False, vocabulary=vocab)
        tf_matrix_POS = vectorizer_pos.fit_transform(naive_bayes_data_train[naive_bayes_labels_train == "pos"]).toarray()  # Vital to make it into dense, if not used it bugs out and does wrong multiply later
        # TFIDF(polarity_1)   
        tfidf_matrix_POS = np.multiply(tf_matrix_POS, idf_train)  # Little trick

        # TF(polarity_2)
        vectorizer_neg = TfidfVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode', use_idf=False, vocabulary=vocab)
        tf_matrix_NEG = vectorizer_neg.fit_transform(naive_bayes_data_train[naive_bayes_labels_train == "neg"]).toarray()  # Vital to make it into dense, if not used it bugs out and does wrong multiply later
        # TFIDF(polarity_2) 
        tfidf_matrix_NEG = np.multiply(tf_matrix_NEG, idf_train)  # Little trick

        # TF of test(polarity_1)
        tf_matrix_test_1 = vectorizer_pos.transform(naive_bayes_data_test).toarray()  # Vital to make it into dense, if not used it bugs out and does wrong multiply later
        # TFIDF of test(polarity_1)
        tfidf_matrix_POS_test = np.multiply(tf_matrix_test_1, idf_train)
        #print(tf_matrix_test_1.shape, idf_train.shape, tfidf_matrix_POS.shape)

        # TF of test(polarity_2)
        tf_matrix_test_2 = vectorizer_neg.transform(naive_bayes_data_test).toarray()  # Vital to make it into dense, if not used it bugs out and does wrong multiply later
        # TFIDF of test(polarity_2)
        tfidf_matrix_NEG_test = np.multiply(tf_matrix_test_2, idf_train)


        tfidf_matrix_POS_test = np.mean(tfidf_matrix_POS_test, axis=1).tolist()  
        tfidf_matrix_NEG_test = np.mean(tfidf_matrix_NEG_test, axis=1).tolist()       

        print(tfidf_matrix_POS_test[0:20])
        print(tfidf_matrix_NEG_test[0:20])

        #print(np.mean(tfidf_matrix_POS_test[1]))
        #print(np.mean(tfidf_matrix_NEG_test[1]))
        quit()

        #test_neg = vectorizer_neg.transform(naive_bayes_data_test).toarray().tolist()

        #print(vocab)
        #print(vectorizer_pos.get_feature_names())
        quit()


        ### vectorizer_pos = TfidfVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode')
        ### feature_selection_p = SelectKBest(score_func=chi2, k=500)
        ### tfidf_matrix_POS = vectorizer_pos.fit_transform(naive_bayes_data_train[naive_bayes_labels_train == "pos"])
        ### tfidf_matrix_POS = feature_selection_p.fit_transform(tfidf_matrix_POS, naive_bayes_labels_train[naive_bayes_labels_train == "pos"])
        ### vectorizer_neg = TfidfVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode')
        ### feature_selection_n = SelectKBest(score_func=chi2, k=500)        
        ### tfidf_matrix_NEG = vectorizer_neg.fit_transform(naive_bayes_data_train[naive_bayes_labels_train == "neg"])
        ### tfidf_matrix_NEG = feature_selection_n.fit_transform(tfidf_matrix_NEG, naive_bayes_labels_train[naive_bayes_labels_train == "neg"])
        

        #icd_pos = IndependentComponentsDistribution.from_samples(tfidf_matrix_POS.toarray(), distributions=NormalDistribution)
        #print(len(list(idp)), tfidf_matrix_POS.shape)
      
        #s1 = State(icd_pos, "s1")
        #HMM_pos = HiddenMarkovModel()
        #HMM_pos.add_state( s1 )
        #HMM_pos.add_transition( HMM_pos.start, s1, 1.0 )
        #HMM_pos.add_transition( s1, HMM_pos.end, 1.0 )  # Make sure to add all transition, else it bugs out!
        #HMM_pos.bake()

        #icd_neg = IndependentComponentsDistribution.from_samples(tfidf_matrix_NEG.toarray(), distributions=NormalDistribution)
        #print(len(list(idp)), tfidf_matrix_POS.shape)
      
        #s1 = State(icd_neg, "s1")
        #HMM_neg = HiddenMarkovModel()
        #HMM_neg.add_state( s1 )
        #HMM_neg.add_transition( HMM_neg.start, s1, 1.0 )
        #HMM_neg.add_transition( s1, HMM_neg.end, 1.0 )  # Make sure to add all transition, else it bugs out!
        #HMM_neg.bake()
        
        ### test_pos = vectorizer_pos.transform(naive_bayes_data_test).toarray().tolist()
        ### test_neg = vectorizer_neg.transform(naive_bayes_data_test).toarray().tolist()
        ### test_pos = feature_selection_p.transform(test_pos)
        ### test_neg = feature_selection_n.transform(test_neg)


        #mean_pos = tfidf_matrix_POS.mean(axis=0).tolist()
        #labels_pos = [["tfidf"] for i in range(len(mean_pos))]
        #mean_neg = tfidf_matrix_NEG.mean(axis=1).tolist()
        #labels_neg = [["tfidf"] for i in range(len(mean_neg))]

        HMM_pos = HiddenMarkovModel.from_samples(IndependentComponentsDistribution, n_components=1, X=mean_pos, labels=labels_pos)  # Baum-Welch of course but irelevant in this scenario
        HMM_neg = HiddenMarkovModel.from_samples(IndependentComponentsDistribution, n_components=1, X=mean_neg, labels=labels_neg)  # Baum-Welch of course but irelevant in this scenario

        test_pos = vectorizer_pos.transform(naive_bayes_data_test).mean(axis=1).tolist()
        test_neg = vectorizer_neg.transform(naive_bayes_data_test).mean(axis=1).tolist()
        for i in range(10):
            print(test_pos[i], test_neg[i])
            print(naive_bayes_labels_test.values[i])
        quit()

        predicted = []
        for i in range(len(test_pos)):
            score_of_pos = test_pos[i]
            score_of_neg = test_neg[i]
            #print(score_of_pos, score_of_neg)

            if HMM_pos.log_probability(score_of_pos) > HMM_neg.log_probability(score_of_neg):  # Forward Algorithm, 'formula' is irelevant anyway in this scenario
                predicted.append("pos")
            elif HMM_pos.log_probability(score_of_pos) < HMM_neg.log_probability(score_of_neg):
                predicted.append("neg")
            else:
                print("OPMEGALOL")
        
        accuracy = metrics.accuracy_score(naive_bayes_labels_test, predicted)
        other_metrics_to_print = metrics.classification_report(naive_bayes_labels_test, predicted, target_names=targetnames, output_dict=False)
        confusion_matrix = metrics.confusion_matrix(naive_bayes_labels_test, predicted)
        print('Exact Accuracy: ', accuracy)
        print(other_metrics_to_print)
        print(confusion_matrix)

        quit()


        #print(naive_bayes_data_train)
        #quit()

        #naive_bayes_data_test = data.sample(frac=0.50, random_state=22).reset_index(drop=True)

        # STEP 1 RUN NAIVE BAYES
        pipeline = Pipeline([ # Optimal
                            ('vect', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode')),  # 1-Gram Vectorizer

                            ('tfidf', TfidfTransformer(use_idf=True)),
                            ('feature_selection', SelectKBest(score_func=chi2, k=2000)),  # Dimensionality Reduction                  
                            ('clf', ComplementNB()),])  
        
        pipeline.fit(naive_bayes_data_train, naive_bayes_labels_train)

        # (3) PREDICT
        predicted = pipeline.predict(naive_bayes_data_test)

        accuracy = metrics.accuracy_score(naive_bayes_labels_test, predicted)
        other_metrics_to_print = metrics.classification_report(naive_bayes_labels_test, predicted, target_names=targetnames, output_dict=False)
        confusion_matrix = metrics.confusion_matrix(naive_bayes_labels_test, predicted)
        print('Exact Accuracy: ', accuracy)
        print(other_metrics_to_print)
        print(confusion_matrix)


        vocab = pipeline.named_steps['vect'].get_feature_names()  # This is the total vocabulary
        selected_indices = pipeline.named_steps['feature_selection'].get_support(indices=True)  # This is the vocabulary after feature selection
        vocab = [vocab[i] for i in selected_indices]

        # Split into test and train and will merge later before running the HMM
        #p1 = multiprocessing.Process(target=Create_Artificial_Labels_to_File, args=(data_train, labels_train, vocab, pipeline))
        #p2 = multiprocessing.Process(target=Create_Artificial_Labels_to_File_Test, args=(data_test, labels_test, vocab, pipeline))
        #p1.start()
        #p2.start()
        quit()

    elif pickle_load == 1:

        all_artifically_labeled_data = pickle.load(open('./Pickled Objects/Artificial_Labels_from_Bayes', 'rb'))
        all_data_corresponding_to_labels = pickle.load(open('./Pickled Objects/Data_corresponding_to_Labels_from_Bayes', 'rb'))

        artifically_labeled_data_test = pickle.load(open('./Pickled Objects/Artificial_Labels_from_Bayes_Test_Set', 'rb'))
        data_corresponding_to_labels_test = pickle.load(open('./Pickled Objects/Data_corresponding_to_Labels_from_Bayes_Test_Set', 'rb'))
        
        golden_truth = pickle.load(open('./Pickled Objects/Artifical_Labels_Golden_Truth', 'rb'))

        print(all_artifically_labeled_data[0:10])
        print(all_data_corresponding_to_labels[0:10])
        quit()

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
    enable_highorder = 0
    enable_pickle_load = 0

    if enable_highorder == 1 and enable_pickle_load == 0:
        set_order = 2

        # Pos    
        second_order_labels = []    
        for seq in pos_artifically_labeled_data:
            tempp = []

            if len(seq) > 0:  # else remains empty
                # first            
                tempp.append("dummy" + seq[0])
                for i in range(set_order-1, len(seq)):
                    # concatenate the names of 2 states
                    tempp.append(seq[i-1] + seq[i])

            second_order_labels.append(tempp)
        pos_artifically_labeled_data = second_order_labels  # Convert the name

        # Neg    
        second_order_labels = []    
        for seq in neg_artifically_labeled_data:
            tempp = []

            if len(seq) > 0:  # else remains empty
                # first            
                tempp.append("dummy" + seq[0])
                for i in range(set_order-1, len(seq)):
                    # concatenate the names of 2 states
                    tempp.append(seq[i-1] + seq[i])

            second_order_labels.append(tempp)
        neg_artifically_labeled_data = second_order_labels  # Convert the name

        # Test    
        second_order_labels = []    
        for seq in artifically_labeled_data_test:
            tempp = []

            if len(seq) > 0:  # else remains empty
                # first            
                tempp.append("dummy" + seq[0])
                for i in range(set_order-1, len(seq)):
                    # concatenate the names of 2 states
                    tempp.append(seq[i-1] + seq[i])

            second_order_labels.append(tempp)
        artifically_labeled_data_test = second_order_labels  # Convert the name

        # HIGH-ORDER
        # print(len(pos_artifically_labeled_data), len(neg_artifically_labeled_data))
        print(pos_artifically_labeled_data[1])
        print(pos_artifically_labeled_data[2])
        print(len(pos_artifically_labeled_data))
        print("NOW NEG")
        print(neg_artifically_labeled_data[1])
        print(neg_artifically_labeled_data[2])
        print(len(neg_artifically_labeled_data))
        count_pos = 0
        count_neg = 0

        # # Check lengths to make sure semi-supervised learning doesn't get enabled (rip it gets enabled anyway)
        # print(len(pos_data_corresponding_to_labels), len(pos_artifically_labeled_data), len(neg_data_corresponding_to_labels), len(neg_artifically_labeled_data))
        # for i in range(len(pos_data_corresponding_to_labels)):
        #     if len(pos_data_corresponding_to_labels[i]) != len(pos_artifically_labeled_data[i]):
        #         print("1")
        #     if None in pos_data_corresponding_to_labels[i] or None in pos_artifically_labeled_data[i]:
        #         print("2")
        # for i in range(len(neg_data_corresponding_to_labels)):
        #     if len(neg_data_corresponding_to_labels[i]) != len(neg_artifically_labeled_data[i]):
        #         print("1")
        #     if None in neg_data_corresponding_to_labels[i] or None in neg_artifically_labeled_data[i]:
        #         print("2")       
        # quit()

        # Shorten the "neg" else it trains for long time compared to "pos" when using emission_pseudocount OR when setting second-order
        for i in range(6000):  # Arbitary number, ~6000+ would include all instances
            #pos_artifically_labeled_data[i] = pos_artifically_labeled_data[i][:-1]
            neg_artifically_labeled_data[i] = neg_artifically_labeled_data[i][:-1]
            neg_data_corresponding_to_labels[i] = neg_data_corresponding_to_labels[i][:-1]           
            #print(pos_artifically_labeled_data[i][0:2])
            count_pos += len(pos_artifically_labeled_data[i])
            count_neg += len(neg_artifically_labeled_data[i])
        print(count_pos/6000.0, count_neg/6000.0)

        # Build Pos Class HMM - !!! state_names should be in alphabetical order
        hmm_supervised_pos = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=6, X=pos_data_corresponding_to_labels, labels=pos_artifically_labeled_data, n_jobs=1, emission_pseudocount=0.5e-05, verbose=True, state_names=["dummyneg", "dummypos", "negneg", "negpos", "posneg", "pospos"])
        print("NEXT HMM")
        # Build Neg Class HMM - !!! state_names should be in alphabetical order
        hmm_supervised_neg = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=6, X=neg_data_corresponding_to_labels, labels=neg_artifically_labeled_data, n_jobs=1, emission_pseudocount=0.5e-05, verbose=True, state_names=["dummyneg", "dummypos", "negneg", "negpos", "posneg", "pospos"])

        #print(hmm_supervised_pos)
        #quit()

        with open('./Pickled Objects/High_Order_HMM_POS', 'wb') as f:
            pickle.dump(hmm_supervised_pos, f)
        with open('./Pickled Objects/High_Order_HMM_NEG', 'wb') as f:
            pickle.dump(hmm_supervised_neg, f)
        with open('./Pickled Objects/High_Order_Test_Set', 'wb') as f:
            pickle.dump(artifically_labeled_data_test, f)

    elif enable_highorder == 1 and enable_pickle_load == 1:
        hmm_supervised_pos = pickle.load(open('./Pickled Objects/High_Order_HMM_POS', 'rb'))
        hmm_supervised_neg = pickle.load(open('./Pickled Objects/High_Order_HMM_NEG', 'rb'))   
        artifically_labeled_data_test = pickle.load(open('./Pickled Objects/High_Order_Test_Set', 'rb')) 

    else:
        # Shorten the "neg" else it trains for long time compared to "pos" when using emission_pseudocount OR when setting second-order
        count_pos = 0
        count_neg = 0
        for i in range(6000):  # Arbitary number, ~6000+ would include all instances
            #pos_artifically_labeled_data[i] = pos_artifically_labeled_data[i][:-1]
            neg_artifically_labeled_data[i] = neg_artifically_labeled_data[i][:-1]
            neg_data_corresponding_to_labels[i] = neg_data_corresponding_to_labels[i][:-1]           
            #print(pos_artifically_labeled_data[i][0:2])
            count_pos += len(pos_artifically_labeled_data[i])
            count_neg += len(neg_artifically_labeled_data[i])
        print(count_pos/6000.0, count_neg/6000.0)

        # Training
        # Build Pos Class HMM - !!! state_names should be in alphabetical order
        hmm_supervised_pos = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=2, X=pos_data_corresponding_to_labels, labels=pos_artifically_labeled_data, emission_pseudocount=0.5e-05, n_jobs=1, state_names=["neg", "pos"])
        print("NEXT HMM")
        # Build Neg Class HMM - !!! state_names should be in alphabetical order
        hmm_supervised_neg = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=2, X=neg_data_corresponding_to_labels, labels=neg_artifically_labeled_data, emission_pseudocount=0.5e-05, n_jobs=1, state_names=["neg", "pos"])       
        # Note: Algorithm used is Baum-Welch


    #print(neg_artifically_labeled_data[0])

    # Testing
    #print(len(pos_artifically_labeled_data), len(neg_artifically_labeled_data))
    #print(hmm_supervised_pos)
    #test_observ = ["realise", "pure"]
    #test_states = ["neg", "neg"]
    #

    transition_proba_matrix_pos = hmm_supervised_pos.dense_transition_matrix()
    transition_proba_matrix_neg = hmm_supervised_neg.dense_transition_matrix()
  
    # !!! Debug the matrix to find which one is which
    # print(transition_proba_matrix_pos)
    # fig, ax1 = plt.subplots()
    # hmm_supervised_pos.plot()
    # plt.show() 
    # quit()
    # # Result :  
    # neg  [0]
    # pos  [1]
    # none-start [2]
    # none-end  [3]
    if enable_highorder == 0:
        mapping = ["neg", "pos", "start", "end"]
    else:
        mapping = ["dummyneg", "dummypos", "negneg", "negpos", "posneg", "pospos", "start", "end"]
    # PLOT HOW THIS VARIABLE AFFECTS PERFORMANCE
    unseen_factor_smoothing = 0.5e-05  # Probability if we stumble upon new unseen observation 
    predicted = []
    test_data_size = len(data_corresponding_to_labels_test)
    count_newunseen = 0
    count_problematic = 0
    empty_sequences = 0

    #print(hmm_supervised_pos.states[1].distribution.parameters[0]["oh"])

    # MATH: score = π(state1) * EmmisionProb(o1|state1) * P(o2|o1) * EmmisionProb(o2|state2) * P(o3|o2) * ...  / SequenceLength

    for k in range(test_data_size):
        current_observations = data_corresponding_to_labels_test[k]
        current_states = artifically_labeled_data_test[k]
        # Debug
        # print(k, len(current_observations))

        if len(current_observations) > 0:         
            sentiment_score_pos = transition_proba_matrix_pos[mapping.index("start"), mapping.index(current_states[0])]  # Transition from start to first state
            sentiment_score_neg = transition_proba_matrix_neg[mapping.index("start"), mapping.index(current_states[0])]  # Transition from start to first state

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

                trans_prob_pos = transition_proba_matrix_pos[current_state_ind, next_state_ind]
                trans_prob_neg = transition_proba_matrix_neg[current_state_ind, next_state_ind]

                sentiment_score_pos = sentiment_score_pos * emissionprob_pos * trans_prob_pos
                sentiment_score_neg = sentiment_score_neg * emissionprob_neg * trans_prob_neg
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

            # Comparison
            if sentiment_score_pos > sentiment_score_neg:
                predicted.append("pos")
            elif sentiment_score_pos < sentiment_score_neg:
                predicted.append("neg")
            else:
                print("NOT ENOUGH TRAINING DATA OR SOMETHING, performing random guessing")
                count_problematic += 1     
                rng = randint(0, 1)
                if rng == 0:
                    predicted.append("pos")
                else:
                    predicted.append("neg")
        
        else:  # Empty Sequence
            print("EMPTY SEQUENCE, performing random guessing")
            empty_sequences += 1
            rng = randint(0, 1)
            if rng == 0:
                predicted.append("pos")
            else:
                predicted.append("neg")

    #seq_count2 = x_test.shape[0]
    #multivariate_3d_test_matrix = np.empty([seq_count2, 1, feature_count])  # Feature Count Remains the same when we vectorize the test set

    Print_Result_Metrics(golden_truth_test, predicted, targetnames, silent_enable_2, time_counter, 0, "HMM "+str(n_order)+"th Order Supervised")

    print("New unseen observations:", count_newunseen, "Problematic Sequences:", count_problematic, "Empty Sequences:", empty_sequences)

    return None

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