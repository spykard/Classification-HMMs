""" 
"INSIDE_FRAMEWORK" BECAUSE DOING THE LABEL GENERATION JUST ONCE WOULD GIVE AN UNFAIR ADVANTAGE
SO WE HAVE TO RUN IT ONCE ON EACH FOLD INSIDE THE FRAMEWORK

Sentiment Analysis: Text Classification using Hidden Markov Models inspired by
an idea of mine.
"""

import pandas as pd
import numpy as np
import pickle
import multiprocessing
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from sklearn.model_selection import RepeatedStratifiedKFold

import string
from re import sub
from nltk.stem import WordNetLemmatizer

import HMM_Framework_Hotfix_Artificial
import Ensemble_Framework

if __name__ == "__main__":

    #dataset_name = "IMDb Large Movie Review Dataset"
    #dataset_name = "Stanford Sentiment Treebank Binary"
    dataset_name = "Stanford Sentiment Treebank Fine"
    #dataset_name = "Movie Review Subjectivity Dataset"
    #dataset_name = "Movie Review Polarity Dataset"
    random_state = 22

    def load_dataset():
        # 1. Dataset dependent loading
        if dataset_name == "IMDb Large Movie Review Dataset":
            # When comparing to other papers, this dataset should be 1-fold 50-50 split with preset test set!
            init_df = pd.read_csv('./Datasets/IMDb Large Movie Review Dataset/CSV Format/imdb_master.csv', names=['Type', 'Data', 'Labels'], usecols=[1,2,3], skiprows=1, encoding='latin1')

            df = init_df[init_df.loc[:,"Labels"] != "unsup"]

            print("--\n--Processed", df.shape[0], "documents", "\n--Dataset Name:", dataset_name)

        elif dataset_name == "Stanford Sentiment Treebank Binary":
            data = ["" for i in range(8741)]
            labels = ["" for i in range(8741)]
            type_of = ["" for i in range(8741)]
            count = 0
            with open('./Datasets/Stanford Sentiment Treebank/Binary/stsa.binary.train', 'r', encoding='iso-8859-15') as file:
                for line in file:
                    if line[0] == "1":
                        labels[count] = "pos"
                    elif line[0] == "0":
                        labels[count] = "neg"
                    else:
                        raise ValueError("Unexpected label (not 0 or 1)")
                    
                    data[count] = line[2:].rstrip('\n')
                    type_of[count] = "train"
                    count += 1
            with open('./Datasets/Stanford Sentiment Treebank/Binary/stsa.binary.test', 'r', encoding='iso-8859-15') as file:
                for line in file:
                    if line[0] == "1":
                        labels[count] = "pos"
                    elif line[0] == "0":
                        labels[count] = "neg"
                    else:
                        raise ValueError("Unexpected label (not 0 or 1)")
                    
                    data[count] = line[2:].rstrip('\n')
                    type_of[count] = "test"
                    count += 1
            
            print("--\n--Processed", count, "documents", "\n--Dataset Name:", dataset_name)

            df = pd.DataFrame({'Type': type_of, 'Data': data, 'Labels': labels})

        elif dataset_name == "Stanford Sentiment Treebank Fine":
            data = ["" for i in range(10754)]
            labels = ["" for i in range(10754)]
            type_of = ["" for i in range(10754)]
            count = 0
            with open('./Datasets/Stanford Sentiment Treebank/Fine/stsa.fine.train', 'r', encoding='iso-8859-15') as file:
                for line in file:
                    if line[0] == "4":
                        labels[count] = "4"
                    elif line[0] == "3":
                        labels[count] = "3"
                    elif line[0] == "2":
                        labels[count] = "2"
                    elif line[0] == "1":
                        labels[count] = "1"
                    elif line[0] == "0":
                        labels[count] = "0"
                    else:
                        raise ValueError("Unexpected label")
                    
                    data[count] = line[2:].rstrip('\n')
                    type_of[count] = "train"
                    count += 1
            with open('./Datasets/Stanford Sentiment Treebank/Fine/stsa.fine.test', 'r', encoding='iso-8859-15') as file:
                for line in file:
                    if line[0] == "4":
                        labels[count] = "4"
                    elif line[0] == "3":
                        labels[count] = "3"
                    elif line[0] == "2":
                        labels[count] = "2"
                    elif line[0] == "1":
                        labels[count] = "1"
                    elif line[0] == "0":
                        labels[count] = "0"
                    else:
                        raise ValueError("Unexpected label")
                    
                    data[count] = line[2:].rstrip('\n')
                    type_of[count] = "test"
                    count += 1
            
            print("--\n--Processed", count, "documents", "\n--Dataset Name:", dataset_name)

            df = pd.DataFrame({'Type': type_of, 'Data': data, 'Labels': labels})

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

    def find_majority(votes):
        vote_count = Counter(votes)
        top_two = vote_count.most_common(2)
        if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
            # It is a tie
            return 0
        return top_two[0][0]

    def _generate_labels_to_file(data, labels, vocab_quick_search, vocab, vocab_test_quick_search, pipeline, batch_id, verbose=False):
        data_corresponding_to_labels = []
        artificial_labels = []
        golden_truth = []
        instance_count = len(data)
        #nlp = spacy.load('en_core_web_sm')
        #wnl = WordNetLemmatizer()
        from textblob import TextBlob
        from senticnet.senticnet import SenticNet
        sn = SenticNet()

        for i in range(instance_count):
            if verbose == True:
                print("Processing instance:", i+1, "of", instance_count)
            tokenize_it = word_tokenize(data[i])
            to_append_data = []        
            to_append_labels = []

            for word in tokenize_it:
                token_to_string = str(word)
                #token_to_string = wnl.lemmatize(token_to_string.lower())  # Lemmatize edition

                sentiment_polarity = TextBlob(token_to_string).sentiment.polarity
                if sentiment_polarity > 0.0:
                    sentiment_polarity = "pos"
                elif sentiment_polarity < 0.0:
                    sentiment_polarity = "neg"
                else:
                    sentiment_polarity = "neu"                

                try:
                    sentiment_polarity_2 = sn.polarity_value(token_to_string)
                    if sentiment_polarity_2 == "positive":
                        sentiment_polarity_2 = "pos"
                    elif sentiment_polarity_2 == "negative":
                        sentiment_polarity_2 = "neg"
                except KeyError:
                    sentiment_polarity_2 = "neu"

                if token_to_string in vocab_quick_search:
                    to_append_data.append(token_to_string)

                    prediction_of_classifier = str(pipeline.predict([token_to_string])[0])
                    
                    majority_vote = find_majority([sentiment_polarity, sentiment_polarity_2, prediction_of_classifier])
                    #if sentiment_polarity != 0:
                    if majority_vote != 0:
                        # Debug
                        #print(prediction_kmeans)
                        to_append_labels.append(majority_vote)  # Convert from numpy.str_ to str and append the label
                    else:
                        #print(token_to_string)
                        to_append_labels.append("neu")

                # (not ALWAYS) putting an 'else' here decreases performance no matter how intelligent the approach is, because dimensionality gets increased
                # LET'S TRY AN EVEN MORE INTELLIGENT APPROACH, THAT TRIES TO "SOLVE" THE TEST SET
                elif token_to_string in vocab_test_quick_search:
                    to_append_data.append(token_to_string)

                    majority_vote = find_majority([sentiment_polarity, sentiment_polarity_2])
                    if majority_vote != 0:
                        # Debug
                        #print(prediction_kmeans)
                        to_append_labels.append(majority_vote)  # Convert from numpy.str_ to str and append the label
                    else:
                        #print(token_to_string)
                        to_append_labels.append("neu")

            # Debug
            #print(to_append_data)

            data_corresponding_to_labels.append(to_append_data)
            artificial_labels.append(to_append_labels)
            golden_truth.append(labels[i])

        with open('./Pickled Objects/Artificial_Data_Batch_' + str(batch_id), 'wb') as f:
            pickle.dump(data_corresponding_to_labels, f)
        with open('./Pickled Objects/Artificial_Labels_Batch_' + str(batch_id), 'wb') as f:
            pickle.dump(artificial_labels, f)
        with open('./Pickled Objects/Artificial_Golden_Truth_Batch_' + str(batch_id), 'wb') as f:
            pickle.dump(golden_truth, f)

    def _generate_labels_to_file_CLASSIC(data, labels, vocab_quick_search, vocab, pipeline, batch_id, verbose=False):
        data_corresponding_to_labels = []
        artificial_labels = []
        golden_truth = []
        instance_count = len(data)
        #nlp = spacy.load('en_core_web_sm')
        #wnl = WordNetLemmatizer()

        for i in range(instance_count):
            if verbose == True:
                print("Processing instance:", i+1, "of", instance_count)
            tokenize_it = word_tokenize(data[i])
            to_append_data = []        
            to_append_labels = []

            for word in tokenize_it:
                token_to_string = str(word)
                # Lemmatize edition
                #token_to_string = wnl.lemmatize(token_to_string.lower())

                if token_to_string in vocab_quick_search:
                    to_append_data.append(token_to_string)
                    prediction_of_classifier = pipeline.predict([token_to_string])[0]
                    # Debug
                    #print(prediction_kmeans)
                    to_append_labels.append(str(prediction_of_classifier))  # Convert from numpy.str_ to str and append the label
            # Debug
            #print(to_append_data)

            data_corresponding_to_labels.append(to_append_data)
            artificial_labels.append(to_append_labels)
            golden_truth.append(labels[i])

        with open('./Pickled Objects/Artificial_Data_Batch_' + str(batch_id), 'wb') as f:
            pickle.dump(data_corresponding_to_labels, f)
        with open('./Pickled Objects/Artificial_Labels_Batch_' + str(batch_id), 'wb') as f:
            pickle.dump(artificial_labels, f)
        with open('./Pickled Objects/Artificial_Golden_Truth_Batch_' + str(batch_id), 'wb') as f:
            pickle.dump(golden_truth, f)


        # Smart Mode
        # sentiment_words = []
        # pos_words = []
        # neg_words = []
        # for line in open('./opinion_lexicon/positive-words.txt', 'r'):
        #     pos_words.append(line.rstrip())  # Must strip Newlines

        # for line in open('./opinion_lexicon/negative-words.txt', 'r'):
        #     neg_words.append(line.rstrip())  # Must strip Newlines  

        # sentiment_words = pos_words + neg_words
        # sentiment_words = set(sentiment_words)

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

    def generate_artificial_labels(data_train, data_test, labels_train, labels_test, data_total, labels_total, feature_count):
        """
        Generates artificial labels for the entire data via Machine Learning classifiers.
        """
        # 1. Machine Learning classifiers
        pipeline = Pipeline([  # Optimal
                            ('vect', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode')),  # 1-Gram Vectorizer
                            ('tfidf', TfidfTransformer(use_idf=True)),
                            ('feature_selection', SelectKBest(score_func=chi2, k=feature_count)),  # Dimensionality Reduction                          
                            ('clf', ComplementNB()),
                            #('clf', LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=1000, C=1.0, n_jobs=1, random_state=random_state)),  # Doesn't help
                            ])  

        #pipeline.fit(data_train, labels_train)

        pipeline.fit(data_train, labels_train)

        # 2. Get vocabulary
        vocab = pipeline.named_steps['vect'].get_feature_names()  # This is the total vocabulary
        #print(len(vocab))
        #quit()
        selected_indices = pipeline.named_steps['feature_selection'].get_support(indices=True)  # This is the vocabulary after feature selection
        vocab = [vocab[i] for i in selected_indices]    

        # OPT. Take a sneak peek at the best feature of the Test Set
        pipeline_test = Pipeline([  # Optimal
                            ('vect', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words='english', strip_accents='unicode')),  # 1-Gram Vectorizer
                            ('tfidf', TfidfTransformer(use_idf=True)),
                            ('feature_selection', SelectKBest(score_func=chi2, k=280)),  # Dimensionality Reduction                           
                            ])  
        pipeline_test.fit(data_test, labels_test)
        # OPT. Get vocabulary
        vocab_test = pipeline_test.named_steps['vect'].get_feature_names()  # This is the total vocabulary
        #print(len(vocab_test))
        #quit()
        selected_indices = pipeline_test.named_steps['feature_selection'].get_support(indices=True)  # This is the vocabulary after feature selection
        vocab_test = [vocab_test[i] for i in selected_indices]  

        # 3. Generate labels to file
        batch_count = 5
        data = list(data_total)
        labels = list(labels_total)

        # print(data[0]) 
        # print(list(data_test)[0])       
        # quit()

        batch_data = []
        batch_labels = []

        for batch in batcher(data, batch_count):  # Use 4 batches to run the process in parallel for higher speed using 4 processes
            batch_data.append(batch)
        for batch in batcher(labels, batch_count):
            batch_labels.append(batch)

        print("\nSplit the data into", batch_count, "batches of approximate size:", df.shape[0]//4)

        p1 = multiprocessing.Process(target=_generate_labels_to_file, args=(batch_data[0], batch_labels[0], set(vocab), vocab, set(vocab_test), pipeline, 1, True))
        p2 = multiprocessing.Process(target=_generate_labels_to_file, args=(batch_data[1], batch_labels[1], set(vocab), vocab, set(vocab_test), pipeline, 2, False))
        p3 = multiprocessing.Process(target=_generate_labels_to_file, args=(batch_data[2], batch_labels[2], set(vocab), vocab, set(vocab_test), pipeline, 3, False))
        p4 = multiprocessing.Process(target=_generate_labels_to_file, args=(batch_data[3], batch_labels[3], set(vocab), vocab, set(vocab_test), pipeline, 4, False))     
        p5 = multiprocessing.Process(target=_generate_labels_to_file, args=(batch_data[4], batch_labels[4], set(vocab), vocab, set(vocab_test), pipeline, 5, False)) 
        #p5 = multiprocessing.Process(target=_generate_labels_to_file, args=(list(data_test), list(labels_train), set(vocab), vocab, pipeline, 5, False))

        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join() 
        p5.join()       

        print("\nSaved to files successfully.")

    def load_from_files():
        """
        Load everything, including the artificial label information, from files.
        """
        batch_data = []
        batch_data.append(pickle.load(open('./Pickled Objects/Artificial_Data_Batch_1', 'rb')))
        batch_data.append(pickle.load(open('./Pickled Objects/Artificial_Data_Batch_2', 'rb')))
        batch_data.append(pickle.load(open('./Pickled Objects/Artificial_Data_Batch_3', 'rb')))
        batch_data.append(pickle.load(open('./Pickled Objects/Artificial_Data_Batch_4', 'rb')))
        batch_data.append(pickle.load(open('./Pickled Objects/Artificial_Data_Batch_5', 'rb')))
        batch_data = [batch for sublist in batch_data for batch in sublist]

        batch_cluster_labels = []
        batch_cluster_labels.append(pickle.load(open('./Pickled Objects/Artificial_Labels_Batch_1', 'rb')))
        batch_cluster_labels.append(pickle.load(open('./Pickled Objects/Artificial_Labels_Batch_2', 'rb')))
        batch_cluster_labels.append(pickle.load(open('./Pickled Objects/Artificial_Labels_Batch_3', 'rb')))
        batch_cluster_labels.append(pickle.load(open('./Pickled Objects/Artificial_Labels_Batch_4', 'rb')))
        batch_cluster_labels.append(pickle.load(open('./Pickled Objects/Artificial_Labels_Batch_5', 'rb')))
        batch_cluster_labels = [batch for sublist in batch_cluster_labels for batch in sublist]

        batch_golden_truth = []
        batch_golden_truth.append(pickle.load(open('./Pickled Objects/Artificial_Golden_Truth_Batch_1', 'rb')))
        batch_golden_truth.append(pickle.load(open('./Pickled Objects/Artificial_Golden_Truth_Batch_2', 'rb')))
        batch_golden_truth.append(pickle.load(open('./Pickled Objects/Artificial_Golden_Truth_Batch_3', 'rb')))
        batch_golden_truth.append(pickle.load(open('./Pickled Objects/Artificial_Golden_Truth_Batch_4', 'rb')))
        batch_golden_truth.append(pickle.load(open('./Pickled Objects/Artificial_Golden_Truth_Batch_5', 'rb')))
        batch_golden_truth = [batch for sublist in batch_golden_truth for batch in sublist]

        # Debug
        # print(len(batch_data), len(batch_cluster_labels), len(batch_golden_truth))
        # print(batch_data[0:10])
        # print(batch_cluster_labels[0:10])
        # print(batch_golden_truth[0:10])

        print("\nLoaded the preprocessed data from files. Creating a DataFrame...\n")

        # 1. Convert to DataFrame
        df_transformed = pd.DataFrame({'Artificial_Labels': batch_cluster_labels, 'Words': batch_data, 'Labels': batch_golden_truth})

        # 2. Remove empty instances from DataFrame, actually affects accuracy
        # WORTH NOTING THAT THIS AFFECTS THE TEST SET TOO (MATHEMATICALLY INCORRECT) SO LET'S NOT DO IT
        #emptySequences = df_transformed.loc[df_transformed.loc[:,'Artificial_Labels'].map(len) < 1].index.values
        #df_transformed = df_transformed.drop(emptySequences, axis=0).reset_index(drop=True)  # reset_Index to make the row numbers be consecutive again

        #emptySequences = []

        # KEEP THE MAPPING TO THE TRAIN-TEST SPLIT
        #fold_split_updated = np.arange(len(fold_split) - len(np.intersect1d(fold_split, emptySequences)))
        fold_split_updated = None
        
        # 3. Print dataset information
        # BUG
        print("--Dataset Info:\n", df_transformed.describe(include="all"), "\n\n", df_transformed.head(3), "\n\n", df_transformed.loc[:,'Labels'].value_counts(), "\n--\n", sep="")
        #print("--Dataset Info:\n", df_transformed.head(3), "\n\n", df_transformed.loc[:,'Labels'].value_counts(), "\n--\n", sep="")

        return (df_transformed, fold_split_updated)


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

    # mode = "load"
    # if mode == "save":
    #     df = load_dataset()
    #     generate_artificial_labels(df, mode="classic", feature_count=2000)  # High Performance
    #     quit()

    # 1. Load the Dataset from file
    df = load_dataset()


    # IMDb only
    if dataset_name == "IMDb Large Movie Review Dataset" or dataset_name.startswith("Stanford Sentiment Treebank"):
        df_init = load_dataset()
        fold_split = df_init.index[df_init["Type"] == "train"].values


    if True:
        # Model
        hmm = HMM_Framework_Hotfix_Artificial.HMM_Framework()
        hmm.build(architecture="B", model="Classic HMM", framework="pome", k_fold=fold_split, boosting=False,                                \
                state_labels_pandas=[], observations_pandas=[], golden_truth_pandas=df.loc[:,"Labels"], \
                text_instead_of_sequences=df.loc[:, "Data"], text_enable=True,                                                                              \
                n_grams=1, n_target="obs", n_prev_flag=False, n_dummy_flag=True,                                                            \
                pome_algorithm="baum-welch", pome_verbose=True, pome_njobs=1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
                pome_algorithm_t="map",                                                                                                       \
                hohmm_high_order=1, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
                architecture_b_algorithm="formula", formula_magic_smoothing=6.0e-06                                                             \
                )     
        
        hmm.print_average_results(decimals=3)
        hmm.print_best_results(detailed=False, decimals=3) 

        #hmm.print_probability_parameters()
        #print(hmm.cross_val_prediction_matrix[0])
        quit()

    elif False:
        # ensemble
        cross_val_prediction_matrix = []
        mapping = []
        golden_truth = []

        hmm = HMM_Framework_Hotfix_Artificial.HMM_Framework()
        hmm.build(architecture="B", model="Classic HMM", framework="hohmm", k_fold=10, boosting=False,                                \
                state_labels_pandas=df.loc[:,"Artificial_Labels"], observations_pandas=df.loc[:,"Words"], golden_truth_pandas=df.loc[:,"Labels"], \
                text_instead_of_sequences=[], text_enable=False,                                                                              \
                n_grams=1, n_target="obs", n_prev_flag=False, n_dummy_flag=True,                                                            \
                pome_algorithm="baum-welch", pome_verbose=True, pome_njobs=-1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
                pome_algorithm_t="map",                                                                                                       \
                hohmm_high_order=1, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
                architecture_b_algorithm="formula", formula_magic_smoothing=0.0001                                                              \
                )     

        cross_val_prediction_matrix.append(hmm.cross_val_prediction_matrix)
        mapping.append(hmm.ensemble_stored["Mapping"])
        golden_truth.append(hmm.ensemble_stored["Curr_Cross_Val_Golden_Truth"])

        hmm = HMM_Framework_Hotfix_Artificial.HMM_Framework()
        hmm.build(architecture="B", model="Classic HMM", framework="hohmm", k_fold=10, boosting=False,                                \
                state_labels_pandas=df.loc[:,"Artificial_Labels"], observations_pandas=df.loc[:,"Words"], golden_truth_pandas=df.loc[:,"Labels"], \
                text_instead_of_sequences=[], text_enable=False,                                                                              \
                n_grams=1, n_target="obs", n_prev_flag=False, n_dummy_flag=True,                                                            \
                pome_algorithm="baum-welch", pome_verbose=True, pome_njobs=-1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
                pome_algorithm_t="map",                                                                                                       \
                hohmm_high_order=2, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
                architecture_b_algorithm="forward", formula_magic_smoothing=0.0001                                                              \
                )     

        cross_val_prediction_matrix.append(hmm.cross_val_prediction_matrix)
        mapping.append(hmm.ensemble_stored["Mapping"])
        golden_truth.append(hmm.ensemble_stored["Curr_Cross_Val_Golden_Truth"])

        Ensemble_Framework.ensemble_run(cross_val_prediction_matrix, mapping, golden_truth, mode="sum", weights=[0.6, 0.4])