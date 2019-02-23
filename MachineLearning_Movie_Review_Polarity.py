'''
Sentiment Analysis: Text Classification using (1) Complement Naive Bayes, (2) k-Nearest Neighbors, (3) Decision Tree, (4) Random Forest, (5) Logistic Regression (Linear), (6) Linear SVM, (7) Stochastic Gradient Descent on SVM, (8) Multi-layer Perceptron
'''

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import itertools
import numpy as np
import pandas as pd
import string
import copy
from re import sub
from collections import defaultdict
from time import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.externals import joblib

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer


cross_validation_best = [0.0, 0.0, "", [], [], 0.0]           # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
cross_validation_all = defaultdict(list)                      # {Name: (Accuracy, F1-score), (Accuracy, F1-score), ...}
cross_validation_average = defaultdict(list)                  # {Name: (Avg(Accuracy), Avg(F1-score)), ...}
time_complexity_average = defaultdict(list)                   # {Name: [Avg(Train+Test_Time)]


def Run_Preprocessing(dataset_name):
    '''    Dataset Dependant Preprocessing    '''

    # 1. Load the Dataset
    dataset = load_files('./Datasets/Movie Review Polarity Dataset/Full Length Reviews version/txt_sentoken', shuffle=False)

    print("--Processed", len(dataset.data), "documents", "\n--Dataset Name:", dataset_name)

    df_dataset = pd.DataFrame({'Labels': dataset.target, 'Data': dataset.data})

    # 2. Remove empty instances from DataFrame, actually affects accuracy
    emptyCells = df_dataset.loc[df_dataset.iloc[:,1] == ''].index.values
    df_dataset = df_dataset.drop(emptyCells, axis=0).reset_index(drop=True)  # Reset_Index to make the row numbers be consecutive again

    # 3. Balance the Dataset by Undersampling
    # mask = df_dataset.loc[:,'Labels'] == "No"
    # df_dataset_to_undersample = df_dataset[mask].sample(n=1718, random_state=22)
    # df_dataset = df_dataset[~mask]
    # df_dataset = pd.concat([df_dataset, df_dataset_to_undersample], ignore_index=True)
    # df_dataset = df_dataset.sample(frac=1, random_state=22).reset_index(drop=True)

    return df_dataset


def Run_Classifier(grid_search_enable, pickle_enable, silent_enable, pipeline, parameters, data_train, data_test, labels_train, labels_test, targetnames, stopwords_complete_lemmatized, model_name):
    '''    Run Classifier with or without Grid Search after Preprocessing is done    '''
    time_counter = time()

    ## GRID SEARCH ON - Search for the Best Parameters
    if grid_search_enable == 1:

        # (1) TRAIN
        grid_go = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)  # Run GridSearch in all Threads (Parallel)
        grid_go = grid_go.fit(data_train, labels_train)
        print('\n- - - - - BEST PARAMETERS - - - - -')
        print(grid_go.best_score_, 'Accuracy')
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, grid_go.best_params_[param_name]))

        print('\n- - - - - DETAILS - - - - -')
        grid_results = grid_go.cv_results_['params']
        for i in range(len(grid_results)):       
            print(i, 'params - %s; mean - %0.10f; std - %0.10f' % (grid_results[i].values(), grid_go.cv_results_['mean_test_score'][i], grid_go.cv_results_['std_test_score'][i]))

        # (2) Model Persistence (Pickle)
        if pickle_enable == 1: joblib.dump(grid_go.best_estimator_, './pickled_models/Classifier.pkl')  

        # (3) PREDICT
        predicted = grid_go.predict(data_test)


    ## GRID SEARCH OFF - Best Parameters are already known
    else:   

        # (1) TRAIN
        pipeline.fit(data_train, labels_train)
        if model_name not in ['(k-Nearest Neighbors)', '(Decision Tree)', '(Random Forest)', '(Multi-layer Perceptron)']: print('\nNumber of Features/Dimension is:', pipeline.named_steps['clf'].coef_.shape[1])
        if model_name in ['(Decision Tree)']: print('\nNumber of Features/Dimension is:', pipeline.named_steps['clf'].n_features_, '| Tree Depth is:', pipeline.named_steps['clf'].tree_.max_depth)
        if model_name in ['(Random Forest)']: print('\nNumber of Features/Dimension is:', pipeline.named_steps['clf'].n_features_)
        if model_name in ['(Multi-layer Perceptron)']: print('\nNumber of Features/Dimension is:', pipeline.named_steps['clf'].coefs_[0].shape[0]) 

        # (2) Model Persistence (Pickle)
        if pickle_enable == 1: joblib.dump(pipeline, './pickled_models/Classifier.pkl') 

        # (3) PREDICT
        predicted = pipeline.predict(data_test)

    Print_Result_Metrics(silent_enable, labels_test, predicted, targetnames, time_counter, 0, model_name)  


def Print_Result_Metrics(silent_enable, labels_test, predicted, targetnames, time_counter, not_new_model, model_name):
    '''    Print Metrics after Training (Testing phase)    '''
    global cross_validation_best, cross_validation_all, time_complexity_average

    # Times
    if not_new_model == 0:
        time_final = time()-time_counter
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
    if not_new_model == 0:  # Lack of this is a fatal Bug; If this flag is 1 we are storing the same model twice
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


def Print_Result_CrossVal_Best(k):
    '''    Print Metrics only of the best result that occured    '''
    global cross_validation_best

    if cross_validation_best[0] > 0.0:
        print("\n" + "- " * 37, end = "")
        Print_Result_Metrics(0, cross_validation_best[3], cross_validation_best[4], None, cross_validation_best[5], 1, cross_validation_best[2] + " best of " + str(k+1) + " Cross Validations")


def Plot_Results(k, dataset_name):
    '''    Plot the Accuracy of all Classifiers in a Graph    '''
    global cross_validation_all, cross_validation_average

    print("Plotting AVERAGES of Cross Validation...")
    for model in cross_validation_all:
        avg = tuple(np.mean(cross_validation_all[model], axis=0))
        cross_validation_average[model] = avg  # Save the average on a global variable
    indices = np.arange(len(cross_validation_average))
    scores_acc = []
    scores_f1 = []
    model_names = []
    for model in cross_validation_average:
        scores_acc.append(cross_validation_average[model][0]) 
        scores_f1.append(cross_validation_average[model][1])
        model_names.append(model)

    # Reverse the items to appear in correct order
    scores_acc.reverse()
    scores_f1.reverse()
    model_names.reverse()

    fig, ax1 = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(left=0.18, top=0.92, bottom=0.08)
    fig.canvas.set_window_title(dataset_name + " - Averages across " + str(k) + "-fold Cross Validation")
    
    p1 = ax1.barh(indices + 0.35, scores_acc, align="center", height=0.35, label="Accuracy (%)", color="navy", tick_label=model_names)    
    p2 = ax1.barh(indices, scores_f1, align="center", height=0.35, label="Accuracy (%)", color="cornflowerblue", tick_label=model_names)
   
    ax1.set_title(dataset_name + " - Averages across " + str(k) + "-fold Cross Validation")
    ax1.set_xlim([0, 1])
    ax1.xaxis.set_major_locator(MaxNLocator(11))
    ax1.xaxis.grid(True, linestyle='--', which="major", color="grey", alpha=.25)
    ax1.legend((p1[0], p2[0]), ("Accuracy", "F1-score"))

    # Right-hand Y-axis
    indices_new = []
    for i in range(0, len(model_names)):  # Trick to print text on the y axis for both bars
        indices_new.append(indices[i])
        indices_new.append(indices[i] + 0.35) 

    ax2 = ax1.twinx()
    ax2.set_yticks(indices_new)
    ax2.set_ylim(ax1.get_ylim())  # Make sure that the limits are set equally on both yaxis so the ticks line up
    ax2.set_yticklabels(x for x in itertools.chain.from_iterable(itertools.zip_longest(scores_f1,scores_acc)) if x)  # Combine two lists in an alternating fashion
    ax2.set_ylabel("Performance")

    plt.show()
    print()


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


### START

# Stopwords
stopwords_complete = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
wnl = WordNetLemmatizer()
stopwords_complete_lemmatized = set([wnl.lemmatize(word) for word in stopwords_complete])
#

np.set_printoptions(precision=10)  # Numpy Precision when Printing

df_dataset = Run_Preprocessing("Movie Review Polarity Dataset")
all_data = df_dataset.loc[:,'Data']
all_labels = df_dataset.loc[:,'Labels']

print("\n--Dataset Info:\n", df_dataset.describe(include="all"), "\n\n", df_dataset.head(), "\n\n", df_dataset.loc[:,'Labels'].value_counts(), "\n--\n", sep="")

# Split using Cross Validation
set_fold = 5
cross_validation_enable = False  # Enable/Disable Flag; if disabled runs the evaluation just once
k_fold = RepeatedStratifiedKFold(5, n_repeats=1, random_state=22)


# Dimensionality Reduction - 4 different ways to pick the best Features 
#   (1) ('feature_selection', SelectKBest(score_func=chi2, k=5000)),  # 0.852 accuracy                    
#   (2) ('feature_selection', TruncatedSVD(n_components=1000)),  # Has Many Issues
#   (3) ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold=-np.inf, max_features=5000)),  # 0.860 accuracy 
#   (4) ('feature_selection', SelectFromModel(estimator=LinearSVC(penalty='l1', dual=False), threshold=-np.inf, max_features=5000)),  # 0.824 accuracy | Technically L1 is better than L2


### (1) LET'S BUILD : Complement Naive Bayes
cross_validation_best = [0.0, 0.0, "", [], [], 0.0]  # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
for k, (train_indexes, test_indexes) in enumerate(k_fold.split(all_data, all_labels)):  # Split must be done before every classifier because generated object gets exhausted (destroyed)
    print("\n--Current Cross Validation Fold:", k+1)

    data_train = all_data.reindex(train_indexes, copy=True, axis=0)
    labels_train = all_labels.reindex(train_indexes, copy=True, axis=0)
    data_test = all_data.reindex(test_indexes, copy=True, axis=0)
    labels_test = all_labels.reindex(test_indexes, copy=True, axis=0)

    # Grid Search On
    pipeline = Pipeline([
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode')),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(min_df=8, stop_words=None, strip_accents='unicode')),],  # 2-Gram Vectorizer
                        )),
                        ('tfidf', TfidfTransformer()),
                        ('feature_selection', SelectKBest(score_func=chi2)),  # Dimensionality Reduction
                        ('clf', ComplementNB()),])  

    parameters = {'tfidf__use_idf': [True, False],
                  'union__transformer_weights': [{'vect1':1.0, 'vect2':1.0},],
                  'union__vect1__max_df': [0.90, 0.80],
                  'union__vect2__max_df': [0.95, 0.85],
                  'union__vect2__ngram_range': [(2, 2)],
                  'feature_selection__k': [100, 500, 1000, 5000, 8000, 14000],} 

    #Run_Classifier(1, 0, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Complement Naive Bayes)')

    # Grid Search Off
    pipeline = Pipeline([ # Optimal
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                        ('feature_selection', SelectKBest(score_func=chi2, k=5000)),  # Dimensionality Reduction                  
                        ('clf', ComplementNB()),])  

    Run_Classifier(0, 0, 1, pipeline, {}, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Complement Naive Bayes)')
    
    if cross_validation_enable == False:
        break  # Disable Cross Validation

Print_Result_CrossVal_Best(k)
###


### (2) LET'S BUILD : k-Nearest Neighbors  //  Noticed that it performs better when a much bigger Dimensionality Reduction is performed
cross_validation_best = [0.0, 0.0, "", [], [], 0.0]  # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
for k, (train_indexes, test_indexes) in enumerate(k_fold.split(all_data, all_labels)):  # Split must be done before every classifier because generated object gets exhausted (destroyed)
    print("\n--Current Cross Validation Fold:", k+1)

    data_train = all_data.reindex(train_indexes, copy=True, axis=0)
    labels_train = all_labels.reindex(train_indexes, copy=True, axis=0)
    data_test = all_data.reindex(test_indexes, copy=True, axis=0)
    labels_test = all_labels.reindex(test_indexes, copy=True, axis=0)

    # Grid Search On
    pipeline = Pipeline([
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode')),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode')),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                        ('feature_selection', SelectKBest(score_func=chi2)),  # Dimensionality Reduction
                        ('clf', KNeighborsClassifier(n_jobs=-1)),])  

    parameters = {'feature_selection__k': [100, 500, 1000, 5000, 8000, 14000],
                  'clf__n_neighbors': [2, 5, 10, 12],} 

    #Run_Classifier(1, 0, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(k-Nearest Neighbors)')

    # Grid Search Off
    pipeline = Pipeline([ # Optimal
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                        ('feature_selection', SelectKBest(score_func=chi2, k=1000)),  # Dimensionality Reduction                  
                        ('clf', KNeighborsClassifier(n_neighbors=2, n_jobs=-1)),])  

    Run_Classifier(0, 0, 1, pipeline, {}, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(k-Nearest Neighbors)')

    if cross_validation_enable == False:
        break  # Disable Cross Validation

Print_Result_CrossVal_Best(k)
###


### (3) LET'S BUILD : Decision Tree  //  Classification trees are used when the target (label) variable is categorical in nature and Regression trees when it's continuous. PRuning is applied through max_depth
cross_validation_best = [0.0, 0.0, "", [], [], 0.0]  # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
for k, (train_indexes, test_indexes) in enumerate(k_fold.split(all_data, all_labels)):  # Split must be done before every classifier because generated object gets exhausted (destroyed)
    print("\n--Current Cross Validation Fold:", k+1)

    data_train = all_data.reindex(train_indexes, copy=True, axis=0)
    labels_train = all_labels.reindex(train_indexes, copy=True, axis=0)
    data_test = all_data.reindex(test_indexes, copy=True, axis=0)
    labels_test = all_labels.reindex(test_indexes, copy=True, axis=0)

    # Grid Search On
    pipeline = Pipeline([
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode')),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode')),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer()),
                        ('feature_selection', SelectKBest(score_func=chi2)),  # Dimensionality Reduction
                        ('clf', DecisionTreeClassifier()),])  

    parameters = {'feature_selection__k': [100, 1000],
                  'clf__max_depth': [20, 25, 30],
                  'clf__min_samples_leaf': [2, 3, 8],
                  'clf__max_features': ['sqrt', None, 100],} 

    #Run_Classifier(1, 0, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Decision Tree)')

    # Grid Search Off
    pipeline = Pipeline([ # Optimal
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                        ('feature_selection', SelectKBest(score_func=chi2, k=1000)),  # Dimensionality Reduction                  
                        ('clf', DecisionTreeClassifier(max_depth=25 , min_samples_leaf=2, max_features=None)),])  

    Run_Classifier(0, 0, 1, pipeline, {}, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Decision Tree)')

    if cross_validation_enable == False:
        break  # Disable Cross Validation

Print_Result_CrossVal_Best(k)
###


### (4) LET'S BUILD : Random Forest  //  Ideal depth can be found from the previous Decision Tree classifier
cross_validation_best = [0.0, 0.0, "", [], [], 0.0]  # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
for k, (train_indexes, test_indexes) in enumerate(k_fold.split(all_data, all_labels)):  # Split must be done before every classifier because generated object gets exhausted (destroyed)
    print("\n--Current Cross Validation Fold:", k+1)

    data_train = all_data.reindex(train_indexes, copy=True, axis=0)
    labels_train = all_labels.reindex(train_indexes, copy=True, axis=0)
    data_test = all_data.reindex(test_indexes, copy=True, axis=0)
    labels_test = all_labels.reindex(test_indexes, copy=True, axis=0)

    # Grid Search On
    pipeline = Pipeline([
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode')),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode')),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer()),
                        ('feature_selection', SelectKBest(score_func=chi2)),  # Dimensionality Reduction
                        ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1)),])  

    parameters = {'feature_selection__k': [1000],
                  'clf__max_depth': [20, 25, 35],
                  'clf__min_samples_leaf': [2],
                  'clf__max_features': ['sqrt', None],} 

    #Run_Classifier(1, 0, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Random Forest)')

    # Grid Search Off
    pipeline = Pipeline([ # Optimal
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                        ('feature_selection', SelectKBest(score_func=chi2, k=1000)),  # Dimensionality Reduction                  
                        ('clf', RandomForestClassifier(n_estimators=100, max_depth=35, min_samples_leaf=2, max_features='sqrt', n_jobs=-1)),])  

    Run_Classifier(0, 0, 1, pipeline, {}, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Random Forest)')

    if cross_validation_enable == False:
        break  # Disable Cross Validation

Print_Result_CrossVal_Best(k)
###


### (5) LET'S BUILD : Logistic Regression (Linear)
cross_validation_best = [0.0, 0.0, "", [], [], 0.0]  # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
for k, (train_indexes, test_indexes) in enumerate(k_fold.split(all_data, all_labels)):  # Split must be done before every classifier because generated object gets exhausted (destroyed)
    print("\n--Current Cross Validation Fold:", k+1)

    data_train = all_data.reindex(train_indexes, copy=True, axis=0)
    labels_train = all_labels.reindex(train_indexes, copy=True, axis=0)
    data_test = all_data.reindex(test_indexes, copy=True, axis=0)
    labels_test = all_labels.reindex(test_indexes, copy=True, axis=0)

    # Grid Search On
    pipeline = Pipeline([
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode')),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode')),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer()),
                        ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold=-np.inf, max_features=5000)),  # Dimensionality Reduction      
                        ('clf', LogisticRegression(penalty='l2', solver='sag', n_jobs=-1)),])

    parameters = {'clf__max_iter': [100],
                  'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],} 

    #Run_Classifier(1, 0, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Logistic Regression)')

    # Grid Search Off
    pipeline = Pipeline([ # Optimal
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer(use_idf=True)),  
                        ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold=-np.inf, max_features=5000)),  # Dimensionality Reduction             
                        ('clf', LogisticRegression(penalty='l2', solver='sag', max_iter=100, C=500, n_jobs=-1, random_state=22)),])  # Sag Solver because it's faster and Liblinear can't even run in Parallel

    Run_Classifier(0, 0, 1, pipeline, {}, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Logistic Regression)')

    if cross_validation_enable == False:
        break  # Disable Cross Validation

Print_Result_CrossVal_Best(k)
###


### (6) LET'S BUILD : Linear SVM
cross_validation_best = [0.0, 0.0, "", [], [], 0.0]  # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
for k, (train_indexes, test_indexes) in enumerate(k_fold.split(all_data, all_labels)):  # Split must be done before every classifier because generated object gets exhausted (destroyed)
    print("\n--Current Cross Validation Fold:", k+1)

    data_train = all_data.reindex(train_indexes, copy=True, axis=0)
    labels_train = all_labels.reindex(train_indexes, copy=True, axis=0)
    data_test = all_data.reindex(test_indexes, copy=True, axis=0)
    labels_test = all_labels.reindex(test_indexes, copy=True, axis=0)

    # Grid Search On
    pipeline = Pipeline([
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode')),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode')),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer()),
                        ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold=-np.inf, max_features=5000)),  # Dimensionality Reduction      
                        ('clf', LinearSVC(penalty='l2', max_iter=1000, dual=True)),])

    parameters = {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'clf__loss': ['squared_hinge'],} 

    #Run_Classifier(1, 0, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Linear SVM)')

    # Grid Search Off
    pipeline = Pipeline([ # Optimal
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer(use_idf=True)),  
                        ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold=-np.inf, max_features=5000)),  # Dimensionality Reduction             
                        ('clf', LinearSVC(penalty='l2', max_iter=1000, C=1, dual=True)),])  # Dual: True for Text/High Feature Count

    Run_Classifier(0, 0, 1, pipeline, {}, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Linear SVM)')

    if cross_validation_enable == False:
        break  # Disable Cross Validation

Print_Result_CrossVal_Best(k)
###

# Don't print, just plot the averages of all models
Plot_Results(set_fold, "Movie Review Polarity Dataset")
quit()

### (7) LET'S BUILD : Stochastic Gradient Descent on SVM
cross_validation_best = [0.0, 0.0, "", [], [], 0.0]  # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
for k, (train_indexes, test_indexes) in enumerate(k_fold.split(all_data, all_labels)):  # Split must be done before every classifier because generated object gets exhausted (destroyed)
    print("\n--Current Cross Validation Fold:", k+1)

    data_train = all_data.reindex(train_indexes, copy=True, axis=0)
    labels_train = all_labels.reindex(train_indexes, copy=True, axis=0)
    data_test = all_data.reindex(test_indexes, copy=True, axis=0)
    labels_test = all_labels.reindex(test_indexes, copy=True, axis=0)

    # Grid Search On
    pipeline = Pipeline([
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode')),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode')),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer()),
                        ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold=-np.inf, max_features=5000)),  # Dimensionality Reduction      
                        ('clf', SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, tol=None, n_jobs=-1)),])

    parameters = {'clf__alpha': [0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06],  # list(10.0 ** -np.arange(1, 7))
                  'clf__tol': [None, 1e-3, 1e-4],}
    
    #Run_Classifier(1, 0, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Stochastic Gradient Descent on SVM)')

    # Grid Search Off
    pipeline = Pipeline([ # Optimal
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer(use_idf=True)),  
                        ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold=-np.inf, max_features=5000)),  # Dimensionality Reduction             
                        ('clf', SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, alpha=0.001, tol=None, n_jobs=-1)),])  # Loss: Hinge means SVM, Log means Logistic Regression

    Run_Classifier(0, 0, 1, pipeline, {}, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Stochastic Gradient Descent on SVM)')

    if cross_validation_enable == False:
        break  # Disable Cross Validation

Print_Result_CrossVal_Best(k)
###


### (8) LET'S BUILD : Multi-layer Perceptron
cross_validation_best = [0.0, 0.0, "", [], [], 0.0]  # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
for k, (train_indexes, test_indexes) in enumerate(k_fold.split(all_data, all_labels)):  # Split must be done before every classifier because generated object gets exhausted (destroyed)
    print("\n--Current Cross Validation Fold:", k+1)

    data_train = all_data.reindex(train_indexes, copy=True, axis=0)
    labels_train = all_labels.reindex(train_indexes, copy=True, axis=0)
    data_test = all_data.reindex(test_indexes, copy=True, axis=0)
    labels_test = all_labels.reindex(test_indexes, copy=True, axis=0)

    # Grid Search On
    pipeline = Pipeline([
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode')),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode')),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer()),
                        ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold=-np.inf, max_features=5000)),  # Dimensionality Reduction      
                        ('clf', MLPClassifier(verbose=False, random_state=22, max_iter=200, solver='sgd', learning_rate='constant', momentum=0.90)),])

    parameters = {'clf__alpha': [0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06],  # list(10.0 ** -np.arange(1, 7))
                  'clf__learning_rate_init': [0.001, 0.01, 0.07],
                  'clf__hidden_layer_sizes': [(100,), (200,50)],}
    
    #Run_Classifier(1, 0, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Multi-layer Perceptron)')

    # Grid Search Off
    pipeline = Pipeline([ # Optimal
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.90, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),
                        ('tfidf', TfidfTransformer(use_idf=True)),  
                        ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold=-np.inf, max_features=5000)),  # Dimensionality Reduction             
                        #('clf', MLPClassifier(verbose=True, hidden_layer_sizes=(200,50), max_iter=400, solver='sgd', learning_rate='adaptive', learning_rate_init=0.60, momentum=0.50, alpha=1e-01)),])
                        ('clf', MLPClassifier(verbose=False, random_state=22, hidden_layer_sizes=(200,50), max_iter=400, solver='sgd', learning_rate='constant', learning_rate_init=0.07, momentum=0.90, alpha=0.001)),])  


    Run_Classifier(0, 0, 1, pipeline, {}, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Multi-layer Perceptron)')

    if cross_validation_enable == False:
        break  # Disable Cross Validation

Print_Result_CrossVal_Best(k)
###