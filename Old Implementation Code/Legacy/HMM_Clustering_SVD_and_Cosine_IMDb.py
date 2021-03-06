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
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

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

from sklearn.preprocessing import Normalizer


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
    df_dataset = df_dataset.sample(frac=0.10, random_state=22).reset_index(drop=True)

    return df_dataset


def Create_Clustered_to_File(data_train, labels_train, f_tuple):
    # Training Data Transform
    clustered_labeled_data = []
    data_corresponding_to_labels = []
    golden_truth = []
    instance_count = len(data_train)
    nlp = spacy.load('en_core_web_sm')

    # some retarded bug
    data_train_new = data_train.tolist()
    labels_train_new = labels_train.tolist()

    for i in range(instance_count):
        print("Currently in instance:", i, "of Train set")
        tokenize_it = word_tokenize(data_train_new[i])
        to_append_labels = []
        to_append_data = []
        for j in tokenize_it:
            #print(i)
            token_to_string = str(j)
            if token_to_string in f_tuple[0]:
                to_append_data.append(token_to_string)
                get_ind = f_tuple[0].index(token_to_string)
                # we can simply directly append the artificial label itself
                prediction_kmeans = f_tuple[1][get_ind]
                #print(prediction_kmeans)
                to_append_labels.append(str(prediction_kmeans))

        clustered_labeled_data.append(to_append_labels)
        data_corresponding_to_labels.append(to_append_data)
        golden_truth.append(labels_train_new[i])


    with open('./Pickled Objects/Clustered_Labels_from_kMeans', 'wb') as f:
        pickle.dump(clustered_labeled_data, f)
    with open('./Pickled Objects/Data_corresponding_to_Labels_from_clustering', 'wb') as f:
        pickle.dump(data_corresponding_to_labels, f)
    with open('./Pickled Objects/Clustered_Labels_Golden_Truth', 'wb') as f:
        pickle.dump(golden_truth, f)

    # # Pos
    # with open('./Pickled Objects/To Train Pos HMM Clustered/Clustered_Labels_from_kMeans', 'wb') as f:
    #     pickle.dump(pos_clustered_labeled_data, f)
    # with open('./Pickled Objects/To Train Pos HMM Clustered/Data_corresponding_to_Labels_from_clustering', 'wb') as f:
    #     pickle.dump(pos_data_corresponding_to_labels, f)
    # # Neg
    # with open('./Pickled Objects/To Train Neg HMM Clustered/Clustered_Labels_from_kMeans', 'wb') as f:
    #     pickle.dump(neg_clustered_labeled_data, f)
    # with open('./Pickled Objects/To Train Neg HMM Clustered/Data_corresponding_to_Labels_from_clustering', 'wb') as f:
    #     pickle.dump(neg_data_corresponding_to_labels, f)
    # #        
    # with open('./Pickled Objects/Artifical_Labels_Golden_Truth', 'wb') as f:
    #     pickle.dump(golden_truth, f)


def Create_Clustered_to_File_Test(data_test, labels_test, f_tuple):
    # Test Data Transform
    clustered_labeled_data_test = []
    data_corresponding_to_labels_test = []
    golden_truth = []
    instance_count = len(data_test)
    nlp = spacy.load('en_core_web_sm')

    # CHANGE THIS TO BE OUTSIDE THE LOOP TO MAKE IT FASTER
    # some retarded bug
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
            if token_to_string in f_tuple[0]:
                to_append_data.append(token_to_string)
                get_ind = f_tuple[0].index(token_to_string)
                # we can simply directly append the artificial label itself
                prediction_kmeans = f_tuple[1][get_ind]
                to_append_labels.append(str(prediction_kmeans))
        
        clustered_labeled_data_test.append(to_append_labels)
        data_corresponding_to_labels_test.append(to_append_data)
        golden_truth.append(labels_test_new[i])

    with open('./Pickled Objects/Clustered_Labels_from_kMeans_Test_Set', 'wb') as f:
        pickle.dump(clustered_labeled_data_test, f)
    with open('./Pickled Objects/Data_corresponding_to_Labels_from_clustering_Test_Set', 'wb') as f:
        pickle.dump(data_corresponding_to_labels_test, f)
    with open('./Pickled Objects/Clustered_Labels_Golden_Truth_Test_Set', 'wb') as f:
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

    from scipy.sparse.linalg import svds

    if pickle_load == 0:
        # STEP 1 CLUSTERING
        # ! Using cosine similarity rather than Euclidean distance is referred to as spherical k-means.
        # (https://www.quora.com/How-can-I-use-cosine-similarity-in-clustering-For-example-K-means-clustering)
        # (https://pypi.org/project/spherecluster/0.1.2/)
        pipeline = Pipeline([ # Optimal
                            ('vect', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode')),  # 1-Gram Vectorizer

                            ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),  # ectorizer results are normalized, which makes KMeans behave as spherical k-means for better results
                            #('feature_selection', TruncatedSVD(n_components=15)),  # Has Many Issues
                            ('feature_selection', SelectKBest(score_func=chi2, k=2400)),  # Dimensionality Reduction                  
                            #('clf', KMeans(n_clusters=75 ,max_iter=1000, verbose=True)),
                            ])  
        
        output = pipeline.fit_transform(data_train, labels_train)  # labels_train is just there for API consistency, None is actually used
        output = output.transpose()

        use_scipy_or_sklearn = 0

        if use_scipy_or_sklearn == 1:

            u, s, v = svds(output, k=200)  # For some stupid reason for our task we need words to be rows not columns for the SVD to produce the correct U*S format
            u_s = u*s
            print(u_s.shape)

        else:
            clf = TruncatedSVD(n_components=2200)
            newtry = clf.fit_transform(output)  # generates U*S
            print(newtry.shape)

        # SVD Results are no normalized, we have to REDO the normalization (https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html)

        normalizer = Normalizer(copy=False)
        normalizer.fit_transform(newtry)  # Seems to work


        word_count = newtry.shape[0]
        cos_sim_dot_matrix = np.zeros((word_count, word_count))
        for i in range(word_count):
            for j in range(word_count):
                cos_sim_dot_matrix[i, j] = np.dot(newtry[i, :], newtry[j, :])

        #print(cos_sim_dot_matrix[1,])


        #quit()


        vocab = pipeline.named_steps['vect'].get_feature_names()  # This is the total vocabulary
        selected_indices = pipeline.named_steps['feature_selection'].get_support(indices=True)  # This is the vocabulary after feature selection
        vocab = [vocab[i] for i in selected_indices]

        # A very interesting way to find the best features, getting exact mapping to features IS IMPOSSIBLE in SVD
        #print(clf.components_[0].argsort()[::-1], len(vocab))
        #best_features = [vocab[i] for i in clf.components_[0].argsort()[::-1]]
        #print(best_features[:10])

        spherical_for_text = 1
        from spherecluster import SphericalKMeans

        # This is Eucledian K-means
        if spherical_for_text == 0:
            clf = KMeans(n_clusters=75, max_iter=1000, verbose=True)
            results = clf.fit(newtry)

            predictions = clf.labels_  # prediction results on the labels, equivalent to fit_predict
        else: # This is Spherical K-means
            clf = SphericalKMeans(n_clusters=75, max_iter=1000, verbose=True)
            results = clf.fit(newtry)

            predictions = clf.labels_  # prediction results on the labels, equivalent to fit_predict

        f_tuple = (vocab, predictions)
        print(f_tuple)

        #print(predictions[vocab.index("fun")], predictions[vocab.index("unconvincing")], predictions[vocab.index("wasted")], predictions[vocab.index("bad")], predictions[vocab.index("masterpiece")], predictions[vocab.index("amazing")], predictions[vocab.index("great")] )
        #quit()

        with open('./Pickled Objects/IMDb word-based SVD-norm k-Means 2000_features mapping list', 'wb') as f:
            pickle.dump(f_tuple, f)

        # accuracy = metrics.accuracy_score(labels_test, predicted)
        # other_metrics_to_print = metrics.classification_report(labels_test, predicted, target_names=targetnames, output_dict=False)
        # confusion_matrix = metrics.confusion_matrix(labels_test, predicted)
        # print('Exact Accuracy: ', accuracy)
        # print(other_metrics_to_print)
        # print(confusion_matrix)

        #vocab = pipeline.named_steps['vect'].get_feature_names()  # This is the total vocabulary
        #selected_indices = pipeline.named_steps['feature_selection'].get_support(indices=True)  # This is the vocabulary after feature selection
        #vocab = [vocab[i] for i in selected_indices]

        # p1 = multiprocessing.Process(target=Create_Artificial_Labels_to_File, args=(data_train, labels_train, vocab, pipeline))
        # p2 = multiprocessing.Process(target=Create_Artificial_Labels_to_File_Test, args=(data_test, labels_test, vocab, pipeline))
        # p1.start()
        # p2.start()
        quit()

    elif pickle_load == 1:

        f_tuple = pickle.load(open('./Pickled Objects/IMDb word-based SVD-norm k-Means 2000_features mapping list', 'rb'))

    create_clusters = 1   
    if create_clusters == 1:  

        #print(pipeline.named_steps['clf'].cluster_centers_.shape)
        #print(pipeline.named_steps['clf'].cluster_centers_[0, 0:20])
        #print(pipeline.named_steps['clf'].labels_)

        p1 = multiprocessing.Process(target=Create_Clustered_to_File, args=(data_train, labels_train, f_tuple))
        p2 = multiprocessing.Process(target=Create_Clustered_to_File_Test, args=(data_test, labels_test, f_tuple))
        p1.start()
        p2.start()
        quit()



    pos_clustered_labeled_data = pickle.load(open('./Pickled Objects/To Train Pos HMM Clustered/Clustered_Labels_from_kMeans', 'rb'))
    pos_data_corresponding_to_labels = pickle.load(open('./Pickled Objects/To Train Pos HMM Clustered/Data_corresponding_to_Labels_from_clustering', 'rb'))
    neg_clustered_labeled_data = pickle.load(open('./Pickled Objects/To Train Neg HMM Clustered/Clustered_Labels_from_kMeans', 'rb'))
    neg_data_corresponding_to_labels = pickle.load(open('./Pickled Objects/To Train Neg HMM Clustered/Data_corresponding_to_Labels_from_clustering', 'rb'))

    clustered_labeled_data_test = pickle.load(open('./Pickled Objects/Clustered_Labels_from_kMeans_Test_Set', 'rb'))
    data_corresponding_to_labels_test = pickle.load(open('./Pickled Objects/Data_corresponding_to_Labels_from_clustering_Test_Set', 'rb'))
    golden_truth_test = pickle.load(open('./Pickled Objects/Clustered_Labels_Golden_Truth_Test_Set', 'rb'))

    # print(len(clustered_labeled_data_test), len(data_corresponding_to_labels_test), len(golden_truth_test))
    # print(pos_clustered_labeled_data[0:2])
    # quit()

        # pos_artifically_labeled_data = pickle.load(open('./Pickled Objects/To Train Pos HMM/Artificial_Labels_from_Bayes', 'rb'))
        # pos_data_corresponding_to_labels = pickle.load(open('./Pickled Objects/To Train Pos HMM/Data_corresponding_to_Labels_from_Bayes', 'rb'))
        # neg_artifically_labeled_data = pickle.load(open('./Pickled Objects/To Train Neg HMM/Artificial_Labels_from_Bayes', 'rb'))
        # neg_data_corresponding_to_labels = pickle.load(open('./Pickled Objects/To Train Neg HMM/Data_corresponding_to_Labels_from_Bayes', 'rb'))

        # artifically_labeled_data_test = pickle.load(open('./Pickled Objects/Artificial_Labels_from_Bayes_Test_Set', 'rb'))
        # data_corresponding_to_labels_test = pickle.load(open('./Pickled Objects/Data_corresponding_to_Labels_from_Bayes_Test_Set', 'rb'))
        # golden_truth_test = pickle.load(open('./Pickled Objects/Artifical_Labels_Golden_Truth_Test_Set', 'rb'))

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
        # for i in range(6000):  # Arbitary number, ~6000+ would include all instances
        #     #pos_artifically_labeled_data[i] = pos_artifically_labeled_data[i][:-1]
        #     neg_artifically_labeled_data[i] = neg_artifically_labeled_data[i][:-1]
        #     neg_data_corresponding_to_labels[i] = neg_data_corresponding_to_labels[i][:-1]           
        #     #print(pos_artifically_labeled_data[i][0:2])
        #     count_pos += len(pos_artifically_labeled_data[i])
        #     count_neg += len(neg_artifically_labeled_data[i])
        # print(count_pos/6000.0, count_neg/6000.0)

        # Build Pos Class HMM - !!! state_names should be in alphabetical order
        hmm_supervised_pos = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=6, X=pos_data_corresponding_to_labels, labels=pos_clustered_labeled_data, n_jobs=1, emission_pseudocount=0.5e-05, verbose=True, state_names=["dummyneg", "dummypos", "negneg", "negpos", "posneg", "pospos"])
        print("NEXT HMM")
        # Build Neg Class HMM - !!! state_names should be in alphabetical order
        hmm_supervised_neg = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=6, X=neg_data_corresponding_to_labels, labels=neg_clustered_labeled_data, n_jobs=1, emission_pseudocount=0.5e-05, verbose=True, state_names=["dummyneg", "dummypos", "negneg", "negpos", "posneg", "pospos"])

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
        # for i in range(6000):  # Arbitary number, ~6000+ would include all instances
        #     #pos_artifically_labeled_data[i] = pos_artifically_labeled_data[i][:-1]
        #     neg_clustered_labeled_data[i] = neg_clustered_labeled_data[i][:-1]
        #     neg_data_corresponding_to_labels[i] = neg_data_corresponding_to_labels[i][:-1]           
        #     #print(pos_artifically_labeled_data[i][0:2])
        #     count_pos += len(pos_clustered_labeled_data[i])
        #     count_neg += len(neg_clustered_labeled_data[i])
        # print(count_pos/6000.0, count_neg/6000.0)

        # Shorten the "neg" else it trains for long time compared to "pos" when using emission_pseudocount OR when setting second-order
        # SHORTEN THE SEQUENCES BY A LOT
        count_pos = 0
        count_neg = 0
        for i in range(6000):  # Arbitary number, ~6000+ would include all instances
            #pos_artifically_labeled_data[i] = pos_artifically_labeled_data[i][:-1]2
            neg_clustered_labeled_data[i] = neg_clustered_labeled_data[i][:-25]
            neg_data_corresponding_to_labels[i] = neg_data_corresponding_to_labels[i][:-25]   
            pos_clustered_labeled_data[i] = pos_clustered_labeled_data[i][:-25]
            pos_data_corresponding_to_labels[i] = pos_data_corresponding_to_labels[i][:-25]                           
            #print(pos_artifically_labeled_data[i][0:2])
            count_pos += len(pos_clustered_labeled_data[i])
            count_neg += len(neg_clustered_labeled_data[i])
        print(count_pos/6000.0, count_neg/6000.0)


        train_or_load = 0
        if train_or_load == 1:
            # DUMMY INSTANCES FOR THE CLUSTERS THAT DONT APPEAR AT ALL, THERE ARE SMARTER SOLUTIONS THAN THIS SUCH AS MAPPING THEM MANUALLY AFTERWARDS WITH 0.0 on a 102x102 matrix
            mapping_pos = set()
            for i in range(len(pos_clustered_labeled_data)):
                mapping_pos.update(pos_clustered_labeled_data[i])
            mapping_neg = set()
            for i in range(len(neg_clustered_labeled_data)):
                mapping_neg.update(neg_clustered_labeled_data[i])

            mapping_pos = sorted(list(mapping_pos))
            mapping_neg = sorted(list(mapping_neg))

            print(len(mapping_pos), len(mapping_neg))

            dummy_instances_pos = []
            for i in range(100):
                if str(i) not in mapping_pos:
                    dummy_instances_pos.append(str(i))
            dummy_instances_neg = []
            for i in range(100):
                if str(i) not in mapping_neg:
                    dummy_instances_neg.append(str(i))

            for j in dummy_instances_pos:
                pos_data_corresponding_to_labels.append(["thisisnevergonnahappen_placeholder_word"])
                pos_clustered_labeled_data.append([j])
            for j in dummy_instances_neg:
                neg_data_corresponding_to_labels.append(["thisisnevergonnahappen_placeholder_word"])
                neg_clustered_labeled_data.append([j])


            # Training
            # Build Pos Class HMM - !!! state_names should be in alphabetical order
            hmm_supervised_pos = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=100, X=pos_data_corresponding_to_labels, verbose=True, labels=pos_clustered_labeled_data, transition_pseudocount=0.5e-05, emission_pseudocount=0.5e-05, n_jobs=1)
            print("NEXT HMM")
            # Build Neg Class HMM - !!! state_names should be in alphabetical order
            hmm_supervised_neg = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=100, X=neg_data_corresponding_to_labels, verbose=True, labels=neg_clustered_labeled_data, transition_pseudocount=0.5e-05, emission_pseudocount=0.5e-05, n_jobs=1)       
            # Note: Algorithm used is Baum-Welch

            with open('./Pickled Objects/Clustered_HMM_POS', 'wb') as f:
                pickle.dump(hmm_supervised_pos, f)
            with open('./Pickled Objects/Clustered_HMM_NEG', 'wb') as f:
                pickle.dump(hmm_supervised_neg, f)

    #print(neg_artifically_labeled_data[0])

    hmm_supervised_pos = pickle.load(open('./Pickled Objects/Clustered_HMM_POS', 'rb'))
    hmm_supervised_neg = pickle.load(open('./Pickled Objects/Clustered_HMM_NEG', 'rb'))  

    # I COULD HAVE, FOR A BETTER PERFORMANCE, mapped the transition matrices to match each other with a length of CLUSTERS+2 (102)
    # But much more complicated
    # # Figure out the mappings for the 2 HMMs since some clusters aren't used at all
    # mapping_pos = set()
    # for i in range(len(pos_clustered_labeled_data)):
    #     mapping_pos.update(pos_clustered_labeled_data[i])
    # mapping_neg = set()
    # for i in range(len(neg_clustered_labeled_data)):
    #     mapping_neg.update(neg_clustered_labeled_data[i])

    # mapping_pos = sorted(list(mapping_pos))
    # mapping_neg = sorted(list(mapping_neg))
    # print(mapping_neg)

    # print(hmm_supervised_pos)
    #test_observ = ["realise", "pure"]
    #test_states = ["neg", "neg"]
    #
    print("LOADED")
    transition_proba_matrix_pos = hmm_supervised_pos.dense_transition_matrix()
    transition_proba_matrix_neg = hmm_supervised_neg.dense_transition_matrix()

    # print(list(transition_proba_matrix_pos[0]))
    # print(list(transition_proba_matrix_neg[0]))
    # quit()
    # !!! Debug the matrix to find which one is which (result: they are sorted with "none-start" and "none-end" at the final 2 spots)
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
        # MAPPING
        mapping = [str(i) for i in range(100)]
        mapping = sorted(mapping)
        mapping.append("start")
        mapping.append("end")
    else:
        mapping = ["i don't know"]

    # PLOT HOW THIS VARIABLE AFFECTS PERFORMANCE
    unseen_factor_smoothing = 0.5e-05  # Probability if we stumble upon new unseen observation 
    predicted = []
    test_data_size = len(data_corresponding_to_labels_test)
    count_newunseen = 0
    count_problematic = 0
    empty_sequences = 0

    #print(hmm_supervised_pos.states[1].distribution.parameters[0]["oh"])

    # MATH: score = ??(state1) * EmmisionProb(o1|state1) * P(o2|o1) * EmmisionProb(o2|state2) * P(o3|o2) * ...  / SequenceLength

    for k in range(test_data_size):
        current_observations = data_corresponding_to_labels_test[k]
        current_states = clustered_labeled_data_test[k]
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
                #print("NOT ENOUGH TRAINING DATA OR SOMETHING, performing random guessing")
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

    print("New unseen observations:", count_newunseen, "Problematic Sequences:", count_problematic, "Empty Sequences:", empty_sequences)

    Print_Result_Metrics(golden_truth_test, predicted, targetnames, silent_enable_2, time_counter, 0, "HMM "+str(n_order)+"th Order Supervised")

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