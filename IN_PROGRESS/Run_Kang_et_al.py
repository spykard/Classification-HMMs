""" 
Sentiment Analysis: Text Classification using Hidden Markov Models inspired by
Kang, M., Ahn, J., & Lee, K. (2018). Opinion mining using ensemble text hidden Markov models for text classification. Expert Systems with Applications, 94, 218-227.
"""

import pandas as pd
import numpy as np
import pickle
import multiprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
from nltk.tokenize import word_tokenize
import HMM_Framework
import Ensemble_Framework


#dataset_name = "IMDb Large Movie Review Dataset"
dataset_name = "Movie Review Subjectivity Dataset"
random_state = 22

def load_dataset():
    # 1. Dataset dependent loading
    if dataset_name == "IMDb Large Movie Review Dataset":
        # When comparing to other papers, this dataset should be 1-fold 50-50 split with preset test set!
        init_df = pd.read_csv('./Datasets/IMDb Large Movie Review Dataset/CSV Format/imdb_master.csv', names=['Type', 'Data', 'Labels'], usecols=[1,2,3], skiprows=1, encoding='latin1')

        df = init_df[init_df.loc[:,"Labels"] != "unsup"]

        print("--\n--Processed", df.shape[0], "documents", "\n--Dataset Name:", dataset_name)

    elif dataset_name == "Movie Review Subjectivity Dataset":
        data = ["" for i in range(10000)]
        labels = ["" for i in range(10000)]
        count = 0
        with open('./Datasets/Movie Review Subjectivity Dataset/plot.tok.gt9.5000', 'r', encoding='iso-8859-15') as file:
            for line in file:
                data[count] = line.rstrip('\n')
                labels[count] = "obj"
                count += 1
        with open('./Datasets/Movie Review Subjectivity Dataset/quote.tok.gt9.5000', 'r', encoding='iso-8859-15') as file:
            for line in file:
                data[count] = line.rstrip('\n')
                labels[count] = "subj"
                count += 1
    
        print("--\n--Processed", count, "documents", "\n--Dataset Name:", dataset_name)

        df = pd.DataFrame({'Data': data, 'Labels': labels})

    # 2. Remove empty instances from DataFrame, actually affects accuracy
    emptySequences = df.loc[df.loc[:,'Data'].map(len) < 1].index.values
    df = df.drop(emptySequences, axis=0).reset_index(drop=True)  # reset_Index to make the row numbers be consecutive again

    # 3. Shuffle the Dataset, just to make sure it's not too perfectly ordered
    if True:
        df = df.sample(frac=0.05, random_state=random_state).reset_index(drop=True)

    # 4. Print dataset information
    print("--Dataset Info:\n", df.describe(include="all"), "\n\n", df.head(3), "\n\n", df.loc[:,'Labels'].value_counts(), "\n--\n", sep="")

    # 5. Balance the Dataset by Undersampling
    if False:
        set_label = "pos"
        set_desired = 75

        mask = df.loc[:,'Labels'] == set_label
        df_todo = df[mask]
        df_todo = df_todo.sample(n=set_desired, random_state=random_state)
        df = pd.concat([df[~mask], df_todo], ignore_index=True)
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df

def _generate_labels_to_file(data, labels, final_tuple, batch_id):
    data_corresponding_to_labels = []
    cluster_labels = []
    golden_truth = []
    instance_count = len(data)
    #nlp = spacy.load('en_core_web_sm')

    for i in range(instance_count):
        print("Processing instance:", i+1, "of", instance_count)
        tokenize_it = word_tokenize(data[i])
        to_append_data = []        
        to_append_labels = []
        for word in tokenize_it:
            token_to_string = str(word)
            if token_to_string in final_tuple[0]:
                to_append_data.append(token_to_string)
                get_index = final_tuple[0].index(token_to_string)
                prediction_kmeans = final_tuple[1][get_index]  # append the label
                # Debug
                #print(prediction_kmeans)
                to_append_labels.append(str(prediction_kmeans))
        # Debug
        #print(to_append_data)

        data_corresponding_to_labels.append(to_append_data)
        cluster_labels.append(to_append_labels)
        golden_truth.append(labels[i])

    with open('./Pickled Objects/Clustering_Data_Batch_' + str(batch_id), 'wb') as f:
        pickle.dump(data_corresponding_to_labels, f)
    with open('./Pickled Objects/Clustering_Labels_Batch_' + str(batch_id), 'wb') as f:
        pickle.dump(cluster_labels, f)
    with open('./Pickled Objects/Clustering_Golden_Truth_Batch_' + str(batch_id), 'wb') as f:
        pickle.dump(golden_truth, f)

def batcher(a, n):
    """
    Generator that yields successive n-sized batches from a; n denotes the number of instances in each batch.
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def generate_cluster_labels(df, spherical_for_text=True):
    """
    Generating cluster labels for the entire data. Uses an advanced SVD and Spherical k-Means approach.
    """
    # 1. SVD
    # Using cosine similarity rather than Euclidean distance is referred to as spherical k-Means.
    # https://www.quora.com/How-can-I-use-cosine-similarity-in-clustering-For-example-K-means-clustering
    # https://pypi.org/project/spherecluster/0.1.2/

    # TODO: TRY MY VECTORIZER
    pipeline = Pipeline([  # Optimal
                        ('vect', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode')),  # 1-Gram Vectorizer
                        ('tfidf', TfidfTransformer(norm='l2', use_idf=True)),  # Vectorizer results are normalized, which makes KMeans behave as spherical k-means for better results
                        ])  
    
    term_sentence_matrix = pipeline.fit_transform(df.loc[:, "Data"], df.loc[:, "Labels"])  # Labels are there just for API consistency, None is actually used
    term_sentence_matrix = term_sentence_matrix.transpose()  # For our task, we don't want the SVD to be performed on the documents but on the words instead; thus we need words to be rows not columns for the SVD to produce the correct U*S format

    vocab = pipeline.named_steps['vect'].get_feature_names()  # This is the overall vocabulary

    svd = TruncatedSVD(n_components=300, algorithm="randomized", random_state=random_state)  # There is no exact perfect number of components. We should aim for variance higher than 0.90
                                                                                              # https://stackoverflow.com/questions/12067446/how-many-principal-components-to-takeh
                                                                                              # https://stackoverflow.com/questions/48424084/number-of-components-trucated-svd
    u_s_matrix = svd.fit_transform(term_sentence_matrix)  # generates U*S
    
    normalizer = Normalizer(norm='l2', copy=False)  # SVD Results are not normalized, we have to REDO the normalization (https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html)
    normalizer.fit_transform(u_s_matrix)  

    print("Singular Value Decomposition (SVD) completed. U*S Shape:", u_s_matrix.shape, "| Explained variance of the SVD step:", svd.explained_variance_ratio_.sum() * 100, "\b%")
    
    # HERE TO DO DOT PRODUCT

    # 2. CLUSTERING
    # This is Eucledian K-means
    if spherical_for_text == False:
        clf = KMeans(n_clusters=20, max_iter=1000, random_state=random_state, verbose=True)
        cluster_labels = clf.fit_predict(u_s_matrix)
        #predictions = clf.labels_  # alternatively could use 'fit' and 'labels_
    # This is Spherical K-means
    else: 
        clf = SphericalKMeans(n_clusters=20, max_iter=1000, random_state=random_state, verbose=True)
        cluster_labels = clf.fit_predict(u_s_matrix)
        #predictions = clf.labels_  # alternatively could use 'fit' and 'labels_  
    
    final_tuple = (vocab, cluster_labels)

    # 3. GENERATE LABELS TO FILE
    batch_count = 4
    data = df.loc[:, "Data"].tolist()
    labels = df.loc[:, "Labels"].tolist()

    batch_data = []
    batch_labels = []

    for batch in batcher(data, batch_count):  # Use 4 batches to run the process in parallel for higher speed using 4 processes
        batch_data.append(batch)
    for batch in batcher(labels, batch_count):
        batch_labels.append(batch)

    print("\nSplit the data into", batch_count, "batches of approximate size:", df.shape[0]//4)

    p1 = multiprocessing.Process(target=_generate_labels_to_file, args=(batch_data[0], batch_labels[0], final_tuple, 1))
    p2 = multiprocessing.Process(target=_generate_labels_to_file, args=(batch_data[1], batch_labels[1], final_tuple, 2))
    p3 = multiprocessing.Process(target=_generate_labels_to_file, args=(batch_data[2], batch_labels[2], final_tuple, 3))
    p4 = multiprocessing.Process(target=_generate_labels_to_file, args=(batch_data[3], batch_labels[3], final_tuple, 4))        
    p1.start()
    p2.start()
    p3.start()
    p4.start()

def load_cluster_labels():
    """
    Load everything from files.
    """
    batch_data = []
    batch_data.append(pickle.load(open('./Pickled Objects/Clustering_Data_Batch_1', 'rb')))
    batch_data.append(pickle.load(open('./Pickled Objects/Clustering_Data_Batch_2', 'rb')))
    batch_data.append(pickle.load(open('./Pickled Objects/Clustering_Data_Batch_3', 'rb')))
    batch_data.append(pickle.load(open('./Pickled Objects/Clustering_Data_Batch_4', 'rb')))
    batch_data = [batch for sublist in batch_data for batch in sublist]


    batch_cluster_labels = []
    batch_cluster_labels.append(pickle.load(open('./Pickled Objects/Clustering_Labels_Batch_1', 'rb')))
    batch_cluster_labels.append(pickle.load(open('./Pickled Objects/Clustering_Labels_Batch_2', 'rb')))
    batch_cluster_labels.append(pickle.load(open('./Pickled Objects/Clustering_Labels_Batch_3', 'rb')))
    batch_cluster_labels.append(pickle.load(open('./Pickled Objects/Clustering_Labels_Batch_4', 'rb')))
    batch_cluster_labels = [batch for sublist in batch_cluster_labels for batch in sublist]

    batch_golden_truth = []
    batch_golden_truth.append(pickle.load(open('./Pickled Objects/Clustering_Golden_Truth_Batch_1', 'rb')))
    batch_golden_truth.append(pickle.load(open('./Pickled Objects/Clustering_Golden_Truth_Batch_2', 'rb')))
    batch_golden_truth.append(pickle.load(open('./Pickled Objects/Clustering_Golden_Truth_Batch_3', 'rb')))
    batch_golden_truth.append(pickle.load(open('./Pickled Objects/Clustering_Golden_Truth_Batch_4', 'rb')))
    batch_golden_truth = [batch for sublist in batch_golden_truth for batch in sublist]



    print(len(batch_data), len(batch_cluster_labels), len(batch_golden_truth))
    print(batch_data[0:10])
    quit()

    pos_data_corresponding_to_labels = pickle.load(open('./Pickled Objects/To Train Pos HMM Clustered/Data_corresponding_to_Labels_from_clustering', 'rb'))
    neg_clustered_labeled_data = pickle.load(open('./Pickled Objects/To Train Neg HMM Clustered/Clustered_Labels_from_kMeans', 'rb'))
    neg_data_corresponding_to_labels = pickle.load(open('./Pickled Objects/To Train Neg HMM Clustered/Data_corresponding_to_Labels_from_clustering', 'rb'))

    clustered_labeled_data_test = pickle.load(open('./Pickled Objects/Clustered_Labels_from_kMeans_Test_Set', 'rb'))
    data_corresponding_to_labels_test = pickle.load(open('./Pickled Objects/Data_corresponding_to_Labels_from_clustering_Test_Set', 'rb'))
    golden_truth_test = pickle.load(open('./Pickled Objects/Clustered_Labels_Golden_Truth_Test_Set', 'rb'))

    return (df_Data, df_Clustering_Labels, df_Golden_Truth)



# MAIN
# HMM_Framework.build
#       General Settings
#       Data
#       Text Scenario
#       n-gram Settings
#       1st Framework Training Settings (High-Order done through the n-grams Settings)
#       1st Framework Prediction Settings (Achitecture A)
#       2nd Framework Training Settings (High-Order done through the 'hohmm_high_order' parameter)
#       Any Framework Prediction Settings (Architecture B)

df = load_dataset()
#generate_cluster_labels(df, spherical_for_text=True)
load_cluster_labels()

quit()

if False:
    # Model
    general_mixture_model_labels = HMM_Framework.general_mixture_model_label_generator(df.loc[:,"Sequences"], df.loc[:,"Labels"])
    hmm = HMM_Framework.HMM_Framework()
    hmm.build(architecture="A", model="General Mixture Model", framework="pome", k_fold=5, boosting=False,                                \
            state_labels_pandas=general_mixture_model_labels, observations_pandas=df.loc[:,"Sequences"], golden_truth_pandas=df.loc[:,"Labels"], \
            text_instead_of_sequences=[], text_enable=False,                                                                              \
            n_grams=1, n_target="both", n_prev_flag=False, n_dummy_flag=False,                                                            \
            pome_algorithm="baum-welch", pome_verbose=False, pome_njobs=1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
            pome_algorithm_t="map",                                                                                                       \
            hohmm_high_order=1, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
            architecture_b_algorithm="forward", formula_magic_smoothing=0.0                                                               \
            )     
    
    hmm.print_average_results(decimals=3)
    hmm.print_best_results(detailed=False, decimals=3) 

elif False:
    #  Model
    #  Just for State-emission HMM, might need to remove the "mix" label during preprocessing.
    hmm = HMM_Framework.HMM_Framework()
    hmm.build(architecture="A", model="State-emission HMM", framework="pome", k_fold=5, boosting=False,                                   \
            state_labels_pandas=df.loc[:,"Sequences"], observations_pandas=df.loc[:,"Sequences"], golden_truth_pandas=df.loc[:,"Labels"], \
            text_instead_of_sequences=[], text_enable=False,                                                                              \
            n_grams=1, n_target="obs", n_prev_flag=False, n_dummy_flag=False,                                                             \
            pome_algorithm="baum-welch", pome_verbose=False, pome_njobs=1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
            pome_algorithm_t="map",                                                                                                       \
            hohmm_high_order=2, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
            architecture_b_algorithm="forward", formula_magic_smoothing=0.0                                                               \
            )   

    hmm.print_average_results(decimals=3)
    hmm.print_best_results(detailed=False, decimals=3) 
