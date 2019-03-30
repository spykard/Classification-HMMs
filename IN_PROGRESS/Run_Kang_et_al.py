""" 
Sentiment Analysis: Text Classification using Hidden Markov Models inspired by
Kang, M., Ahn, J., & Lee, K. (2018). Opinion mining using ensemble text hidden Markov models for text classification. Expert Systems with Applications, 94, 218-227.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
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
                data[count] = line
                labels[count] = "obj"
                count += 1
        with open('./Datasets/Movie Review Subjectivity Dataset/quote.tok.gt9.5000', 'r', encoding='iso-8859-15') as file:
            for line in file:
                data[count] = line
                labels[count] = "subj"
                count += 1
    
        print("--\n--Processed", count, "documents", "\n--Dataset Name:", dataset_name)

        df = pd.DataFrame({'Data': data, 'Labels': labels})

    # 2. Remove empty instances from DataFrame, actually affects accuracy
    emptySequences = df.loc[df.loc[:,'Data'].map(len) < 1].index.values
    df = df.drop(emptySequences, axis=0).reset_index(drop=True)  # reset_Index to make the row numbers be consecutive again

    # 3. Shuffle the Dataset, just to make sure it's not too perfectly ordered
    if True:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 4. Print dataset information
    print("--Dataset Info:\n", df.describe(include="all"), "\n\n", df.head(3), "\n\n", df.loc[:,'Labels'].value_counts(), "\n--\n", sep="")

    # 5. Balance the Dataset by Undersampling
    if False:
        set_label = "neu"
        set_desired = 75

        mask = df.loc[:,'Labels'] == set_label
        df_todo = df[mask]
        df_todo = df_todo.sample(n=set_desired, random_state=random_state)
        df = pd.concat([df[~mask], df_todo], ignore_index=True)
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df

def generate_cluster_labels(df, spherical_for_text=True):
    """
    Generating cluster labels for the entire data. Uses an advanced SVD and Spherical k-Means approach.
    """
    # 1. CLUSTERING
    # Using cosine similarity rather than Euclidean distance is referred to as spherical k-means.
    # https://www.quora.com/How-can-I-use-cosine-similarity-in-clustering-For-example-K-means-clustering
    # https://pypi.org/project/spherecluster/0.1.2/

    # TODO: TRY MY VECTORIZER
    pipeline = Pipeline([  # Optimal
                        ('vect', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode')),  # 1-Gram Vectorizer
                        ('tfidf', TfidfTransformer(norm='l2', use_idf=True)),  # Vectorizer results are normalized, which makes KMeans behave as spherical k-means for better results
                        ])  
    
    term_sentence_matrix = pipeline.fit_transform(df.loc[:, "Data"], df.loc[:, "Labels"])  # Labels are there just for API consistency, None is actually used
    term_sentence_matrix = term_sentence_matrix.transpose()  # For some stupid reason, for our task we need words to be rows not columns for the SVD to produce the correct U*S format

    svd = TruncatedSVD(n_components=300, algorithm="randomized", random_state=random_state)  # There is no exact perfect number of components. We should aim for variance higher than 0.90
                                                                                              # https://stackoverflow.com/questions/12067446/how-many-principal-components-to-takeh
                                                                                              # https://stackoverflow.com/questions/48424084/number-of-components-trucated-svd
    u_s_matrix = svd.fit_transform(term_sentence_matrix)  # generates U*S
    
    normalizer = Normalizer(norm='l2', copy=False)  # SVD Results are not normalized, we have to REDO the normalization (https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html)
    normalizer.fit_transform(u_s_matrix)  

    print("Singular Value Decomposition (SVD) completed. U*S Shape:", u_s_matrix.shape, "| Explained variance of the SVD step:", svd.explained_variance_ratio_.sum())
    
    # HERE TO DO DOT PRODUCT


    # This is Eucledian K-means
    if spherical_for_text == False:
        clf = KMeans(n_clusters=75, max_iter=1000, verbose=True)
        cluster_labels = clf.fit_predict(u_s_matrix)
        #predictions = clf.labels_  # alternatively could use 'fit' and 'labels_
    # This is Spherical K-means
    else: 
        clf = SphericalKMeans(n_clusters=75, max_iter=1000, verbose=True)
        cluster_labels = clf.fit_predict(u_s_matrix)
        #predictions = clf.labels_  # alternatively could use 'fit' and 'labels_  
    
    print(len(results))
    quit()


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
generate_cluster_labels(df, spherical_for_text=True)
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
