""" 
Sentiment Analysis: Text Classification using Hidden Markov Models inspired by
an idea of mine.
"""

import pandas as pd
import numpy as np
import pickle
import multiprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from nltk.tokenize import word_tokenize

import string
from re import sub
from nltk.stem import WordNetLemmatizer

import HMM_Framework
import Ensemble_Framework


#dataset_name = "IMDb Large Movie Review Dataset"
#dataset_name = "Movie Review Subjectivity Dataset"
dataset_name = "Movie Review Polarity Dataset"
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

    elif dataset_name == "Movie Review Polarity Dataset":
        data = ["" for i in range(10662)]
        labels = ["" for i in range(10662)]
        count = 0
        with open('./Datasets/Movie Review Polarity Dataset/Sentence Polarity version/rt-polaritydata/rt-polarity.neg', 'r', encoding='iso-8859-15') as file:
            for line in file:
                data[count] = line.rstrip('\n')
                labels[count] = "neg"
                count += 1
        with open('./Datasets/Movie Review Polarity Dataset/Sentence Polarity version/rt-polaritydata/rt-polarity.pos', 'r', encoding='iso-8859-15') as file:
            for line in file:
                data[count] = line.rstrip('\n')
                labels[count] = "pos"
                count += 1
    
        print("--\n--Processed", count, "documents", "\n--Dataset Name:", dataset_name)

        df = pd.DataFrame({'Data': data, 'Labels': labels})

    # 2. Remove empty instances from DataFrame, actually affects accuracy
    emptySequences = df.loc[df.loc[:,'Data'].map(len) < 1].index.values
    df = df.drop(emptySequences, axis=0).reset_index(drop=True)  # reset_Index to make the row numbers be consecutive again

    # 3. Shuffle the Dataset, just to make sure it's not too perfectly ordered
    if True:
        df = df.sample(frac=1., random_state=random_state).reset_index(drop=True)

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

def _generate_labels_to_file(data, labels, final_tuple, batch_id, verbose=False):
    data_corresponding_to_labels = []
    cluster_labels = []
    golden_truth = []
    instance_count = len(data)
    #nlp = spacy.load('en_core_web_sm')

    for i in range(instance_count):
        if verbose == True:
            print("Processing instance:", i+1, "of", instance_count)
        tokenize_it = word_tokenize(data[i])
        to_append_data = []        
        to_append_labels = []
        for word in tokenize_it:
            token_to_string = str(word)
            if token_to_string in final_tuple[0]:
                to_append_data.append(token_to_string)
                get_index = final_tuple[1].index(token_to_string)
                prediction_kmeans = final_tuple[2][get_index]  # append the label
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

class LemmaTokenizer(object):
    '''    Override SciKit's default Tokenizer    '''
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        # This punctuation remover has the best Speed Performance
        self.translator = str.maketrans('','', sub('\'', '', string.punctuation))
    def __call__(self, doc):
        # return [self.wnl.lemmatize(t.lower()) for t in word_tokenize(doc)]
        temp = []
        for t in word_tokenize(doc):
            x = t.translate(self.translator) 
            if x != '': temp.append(self.wnl.lemmatize(x.lower())) 

        return temp

def generate_artificial_labels(df, mode, n_components, cosine_sim_flag=False, cluster_count=100):
    """
    Generates artificial labels for the entire data via Machine Learning classifiers.
    """
    # 1. Machine Learning classifiers
    pipeline = Pipeline([  # Optimal
                        ('vect', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                        ('tfidf', TfidfTransformer(use_idf=True)),
                        ('clf', ComplementNB()),
                        ])  
    
    term_sentence_matrix = pipeline.fit_transform(df.loc[:, "Data"], df.loc[:, "Labels"])  # Labels are there just for API consistency, None is actually used
    term_sentence_matrix = term_sentence_matrix.transpose()  # For our task, we don't want the SVD to be performed on the documents but on the words instead; thus we need words to be rows not columns for the SVD to produce the correct U*S format

    vocab = pipeline.named_steps['vect'].get_feature_names()  # This is the overall vocabulary

    svd = TruncatedSVD(n_components=n_components, algorithm="randomized", random_state=random_state)  # There is no exact perfect number of components. We should aim for variance higher than 0.90
                                                                                                      # https://stackoverflow.com/questions/12067446/how-many-principal-components-to-takeh
                                                                                                      # https://stackoverflow.com/questions/48424084/number-of-components-trucated-svd
    u_s_matrix = svd.fit_transform(term_sentence_matrix)  # generates U*S
    
    normalizer = Normalizer(norm='l2', copy=False)  # SVD Results are not normalized, we have to REDO the normalization (https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html)
    normalizer.fit_transform(u_s_matrix)

    print("Singular Value Decomposition (SVD) completed. U*S Shape:", u_s_matrix.shape, "| Explained variance of the SVD step:", svd.explained_variance_ratio_.sum() * 100, "\b%")

    # Cosine Similarity  
    if cosine_sim_flag == True:       
        u_s_matrix = cosine_similarity(u_s_matrix, u_s_matrix)

    # 2. CLUSTERING
    # This is Eucledian K-means
    if mode == "classic":
        clf = KMeans(n_clusters=cluster_count, max_iter=1000, random_state=random_state, verbose=True)
        cluster_labels = clf.fit_predict(u_s_matrix)
        #predictions = clf.labels_  # alternatively could use 'fit' and 'labels_
    # This is Spherical K-means
    elif mode == "spherical": 
        clf = SphericalKMeans(n_clusters=cluster_count, max_iter=1000, random_state=random_state, verbose=True)
        cluster_labels = clf.fit_predict(u_s_matrix)
        #predictions = clf.labels_  # alternatively could use 'fit' and 'labels_  
    elif mode == "matlab":        
        matrix = matlab.double(u_s_matrix.tolist())
        eng = matlab.engine.start_matlab()
        output = eng.kmeans(matrix, cluster_count, 'MaxIter', 1000.0, 'Distance', 'cosine', nargout=1)
        cluster_labels = [int(x[0]) for x in output]
    
    print(cluster_labels)

    final_tuple = (set(vocab), vocab, cluster_labels)  # [0] vocab as a set for fast search inside it, [1] vocab as a mapping, [2] cluster labels

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

    p1 = multiprocessing.Process(target=_generate_labels_to_file, args=(batch_data[0], batch_labels[0], final_tuple, 1, True))
    p2 = multiprocessing.Process(target=_generate_labels_to_file, args=(batch_data[1], batch_labels[1], final_tuple, 2, False))
    p3 = multiprocessing.Process(target=_generate_labels_to_file, args=(batch_data[2], batch_labels[2], final_tuple, 3, False))
    p4 = multiprocessing.Process(target=_generate_labels_to_file, args=(batch_data[3], batch_labels[3], final_tuple, 4, False))        
    p1.start()
    p2.start()
    p3.start()
    p4.start()

    print("\nSaved to files successfully.")

def load_from_files():
    """
    Load everything, including the clustering information, from files.
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

    # Debug
    # print(len(batch_data), len(batch_cluster_labels), len(batch_golden_truth))
    # print(batch_data[0:10])
    # print(batch_cluster_labels[0:10])
    # print(batch_golden_truth[0:10])

    print("\nLoaded the preprocessed (and clustered) data from files. Creating a DataFrame...\n")

    # 1. Convert to DataFrame
    df_transformed = pd.DataFrame({'Clustering_Labels': batch_cluster_labels, 'Words': batch_data, 'Labels': batch_golden_truth})

    # 2. Remove empty instances from DataFrame, actually affects accuracy
    emptySequences = df_transformed.loc[df_transformed.loc[:,'Clustering_Labels'].map(len) < 1].index.values
    df_transformed = df_transformed.drop(emptySequences, axis=0).reset_index(drop=True)  # reset_Index to make the row numbers be consecutive again
    
    # 3. Print dataset information
    # BUG
    #print("--Dataset Info:\n", df_transformed.describe(include="all"), "\n\n", df_transformed.head(3), "\n\n", df_transformed.loc[:,'Labels'].value_counts(), "\n--\n", sep="")
    print("--Dataset Info:\n", df_transformed.head(3), "\n\n", df_transformed.loc[:,'Labels'].value_counts(), "\n--\n", sep="")

    return df_transformed


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

mode = "load"
if mode == "save":
    df = load_dataset()
    generate_artificial_labels(df, mode="matlab", n_components=700, cosine_sim_flag=False, cluster_count=60)  # High Performance
    #df = load_from_files()
    quit()
elif mode == "load":
    df = load_from_files()

# IMDb only
if dataset_name == "IMDb Large Movie Review Dataset":
    df_init = load_dataset()
    fold_split = df_init.index[df_init["Type"] == "train"].values


if False:
    # Model
    hmm = HMM_Framework.HMM_Framework()
    hmm.build(architecture="B", model="Classic HMM", framework="pome", k_fold=10, boosting=False,                                \
            state_labels_pandas=df.loc[:,"Clustering_Labels"], observations_pandas=df.loc[:,"Words"], golden_truth_pandas=df.loc[:,"Labels"], \
            text_instead_of_sequences=[], text_enable=False,                                                                              \
            n_grams=2, n_target="obs", n_prev_flag=False, n_dummy_flag=True,                                                            \
            pome_algorithm="baum-welch", pome_verbose=True, pome_njobs=-1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
            pome_algorithm_t="map",                                                                                                       \
            hohmm_high_order=1, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
            architecture_b_algorithm="formula", formula_magic_smoothing=0.0005                                                              \
            )     
    
    hmm.print_average_results(decimals=3)
    hmm.print_best_results(detailed=False, decimals=3) 

    #hmm.print_probability_parameters()
    #print(hmm.cross_val_prediction_matrix[0])

elif True:
    # ensemble
    cross_val_prediction_matrix = []
    mapping = []
    golden_truth = []

    hmm = HMM_Framework.HMM_Framework()
    hmm.build(architecture="B", model="Classic HMM", framework="pome", k_fold=10, boosting=False,                                \
            state_labels_pandas=df.loc[:,"Clustering_Labels"], observations_pandas=df.loc[:,"Words"], golden_truth_pandas=df.loc[:,"Labels"], \
            text_instead_of_sequences=[], text_enable=False,                                                                              \
            n_grams=1, n_target="obs", n_prev_flag=False, n_dummy_flag=True,                                                            \
            pome_algorithm="baum-welch", pome_verbose=True, pome_njobs=-1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
            pome_algorithm_t="map",                                                                                                       \
            hohmm_high_order=1, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
            architecture_b_algorithm="formula", formula_magic_smoothing=0.0005                                                              \
            )     

    cross_val_prediction_matrix.append(hmm.cross_val_prediction_matrix)
    mapping.append(hmm.ensemble_stored["Mapping"])
    golden_truth.append(hmm.ensemble_stored["Curr_Cross_Val_Golden_Truth"])

    hmm = HMM_Framework.HMM_Framework()
    hmm.build(architecture="B", model="Classic HMM", framework="hohmm", k_fold=10, boosting=False,                                \
            state_labels_pandas=df.loc[:,"Clustering_Labels"], observations_pandas=df.loc[:,"Words"], golden_truth_pandas=df.loc[:,"Labels"], \
            text_instead_of_sequences=[], text_enable=False,                                                                              \
            n_grams=1, n_target="obs", n_prev_flag=False, n_dummy_flag=True,                                                            \
            pome_algorithm="baum-welch", pome_verbose=True, pome_njobs=-1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
            pome_algorithm_t="map",                                                                                                       \
            hohmm_high_order=1, hohmm_smoothing=1.5, hohmm_synthesize=False,                                                              \
            architecture_b_algorithm="formula", formula_magic_smoothing=0.0005                                                              \
            )     

    cross_val_prediction_matrix.append(hmm.cross_val_prediction_matrix)
    mapping.append(hmm.ensemble_stored["Mapping"])
    golden_truth.append(hmm.ensemble_stored["Curr_Cross_Val_Golden_Truth"])

    Ensemble_Framework.ensemble_run(cross_val_prediction_matrix, mapping, golden_truth, mode="sum", weights=[0.6, 0.4])