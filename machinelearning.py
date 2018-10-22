'''
Sentiment Analysis: Text Classification using (1) Complement Naive Bayes, (2) k-Nearest Neighbors, (3) Decision Tree, (4) Random Forest, (5) Linear Regression, (6) Logistic Regression, (7) Linear SVM, (8) Stochastic Gradient Descent on SVM, (9) Multi-layer Perceptron
'''

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
from sklearn.datasets import load_files

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.externals import joblib

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import string
import copy
from re import sub

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


cross_validation_best = [0.000, "", [], []]  # [Score, Model Name, Actual Labels, Predicted]
all_models_accuracy = []  # [(Score, Model Name)]  To show comparison in a Graph

def Run_Preprocessing(dataset_name):
    '''    Dataset Dependant Preprocessing    '''

    dataset = load_files('./Datasets/Movie Review Polarity Dataset/txt_sentoken', shuffle=False)

    print("--Processed", len(dataset.data), "documents", "\n--Dataset Name:", dataset_name)

    df_dataset = pd.DataFrame({'Label': dataset.target, 'Data': dataset.data})

    # Remove empty instances from DataFrame
    emptyCells = df_dataset.loc[df_dataset.iloc[:,1] == ''].index.values
    df_dataset = df_dataset.drop(emptyCells, axis=0).reset_index(drop=True)  # Reset_Index to make the row numbers be consecutive again

    return df_dataset


def Run_Classifier(grid_search_enable, pickle_enable, silent_enable, pipeline, parameters, data_train, data_test, labels_train, labels_test, targetnames, stopwords_complete_lemmatized, model_name):
    '''    Run Classifier with or without Grid Search after Preprocessing is done    '''

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
        if model_name not in ['(k-Nearest Neighbors)', '(MultiLayer Perceptron)']: print('\nNumber of Features/Dimension is:', pipeline.named_steps['clf'].coef_.shape[1])
        if model_name in ['(Decision Tree)']: print('\nNumber of Features/Dimension is:', pipeline.named_steps['clf'].n_features_)

        # (2) Model Persistence (Pickle)
        if pickle_enable == 1: joblib.dump(pipeline, './pickled_models/Classifier.pkl') 

        # (3) PREDICT
        predicted = pipeline.predict(data_test)

    Print_Result_Metrics(silent_enable, labels_test, predicted, targetnames, model_name)  


def Print_Result_Metrics(silent_enable, labels_test, predicted, targetnames, model_name):
    '''    Print Metrics after Training etc.    '''
    global cross_validation_best

    accuracy = metrics.accuracy_score(labels_test, predicted)
    if silent_enable == 0:
        print('\n- - - - - RESULT METRICS -', model_name, '- - - - -')
        print('Exact Accuracy: ', accuracy)
        print(metrics.classification_report(labels_test, predicted, target_names=targetnames))
        print(metrics.confusion_matrix(labels_test, predicted))

    if accuracy > cross_validation_best[0]:
        cross_validation_best[0] = accuracy
        cross_validation_best[1] = model_name
        cross_validation_best[2] = labels_test
        cross_validation_best[3] = predicted 


def Print_Result_Best():
    '''    Print Metrics only of the best result that occured    '''
    global cross_validation_best
    global all_models_accuracy

    if cross_validation_best[0] > 0.000:
        all_models_accuracy.append((cross_validation_best[0], cross_validation_best[1])) 
        print("\n\n" + "- " * 37, end = "")
        Print_Result_Metrics(0, cross_validation_best[2], cross_validation_best[3], None, cross_validation_best[1] + " best of " + str(k+1) + " Cross Validations")


def Plot_Results(dataset_name):
    '''    Plot the Accuracy of all Classifiers in a Graph    '''
    global all_models_accuracy

    indices = np.arange(len(all_models_accuracy))
    scores = [x[0] for x in reversed(all_models_accuracy)]  # Reverse of the List to appear in correct order
    model_names = [x[1][1:-1] for x in reversed(all_models_accuracy)]  # Reverse of the List to appear in correct order

    fig, ax1 = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(left=0.18, top=0.92, bottom=0.08)
    fig.canvas.set_window_title(dataset_name + " - Score")
    
    ax1.barh(indices, scores, align="center", height=0.35, label="Accuracy (%)", color="navy", tick_label=model_names)
    ax1.set_title(dataset_name + " - Score")
    ax1.set_xlim([0, 1])
    ax1.xaxis.set_major_locator(MaxNLocator(11))
    ax1.xaxis.grid(True, linestyle='--', which="major", color="grey", alpha=.25)

    # Right-hand Y-axis
    ax2 = ax1.twinx()
    ax2.set_yticks(indices)
    ax2.set_ylim(ax1.get_ylim())  # Make sure that the limits are set equally on both yaxis so the ticks line up
    ax2.set_yticklabels(scores)
    ax2.set_ylabel('Test Accuracy')

    plt.show()


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

stopwords_complete = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
wnl = WordNetLemmatizer()
stopwords_complete_lemmatized = set([wnl.lemmatize(word) for word in stopwords_complete])

np.set_printoptions(precision=10)  # Numpy Precision when Printing

df_dataset = Run_Preprocessing("Movie Review Polarity Dataset")
all_data = df_dataset.iloc[:,1]
all_labels = df_dataset.iloc[:,0]

print("\n--Dataset Info:\n", df_dataset.describe(include="all"), "\n\n", df_dataset.head(), "\n")

# Split using Cross Validation
k_fold = RepeatedStratifiedKFold(4, n_repeats=1, random_state=22)
k_fold_data = k_fold.split(all_data, all_labels)


# Dimensionality Reduction - 4 different ways to pick the best Features 
#   (1) ('feature_selection', SelectKBest(score_func=chi2, k=5000)),                    
#   (2) ('feature_selection', TruncatedSVD(n_components=1000)),  # Has Many Issues
#   (3) ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold=-np.inf, max_features=5000)),
#   (4) ('feature_selection', SelectFromModel(estimator=LinearSVC(penalty='l1', dual=False), threshold=-np.inf, max_features=5000)),  # Technically L1 is better than L2


### (1) LET'S BUILD : Complement Naive Bayes

for k, (train_indexes, test_indexes) in enumerate(k_fold_data):
    print("\n--Current Cross Validation Fold:", k)

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

    #Run_Classifier(0, 0, 1, pipeline, {}, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Complement Naive Bayes)')
    break  # Disable Cross Validation

Print_Result_Best()  # Best Cross Validation
###


### (2) LET'S BUILD : k-Nearest Neighbors  //  Noticed that it performs better when a much bigger Dimensionality Reduction is performed
cross_validation_best = [0.000, "", [], []]
for k, (train_indexes, test_indexes) in enumerate(k_fold_data):
    print("\n--Current Cross Validation Fold:", k)

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
    break  # Disable Cross Validation

Print_Result_Best()  # Best Cross Validation
###


### (3) LET'S BUILD : Decision Tree  //  Classification trees are used when the target (label) variable is categorical in nature and Regression trees when it's continuous
cross_validation_best = [0.000, "", [], []]
for k, (train_indexes, test_indexes) in enumerate(k_fold_data):
    print("\n--Current Cross Validation Fold:", k)

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

    parameters = {'feature_selection__k': [100, 500, 1000],
                  'clf__max_features': ['sqrt', 'log2', None]} 

    Run_Classifier(1, 0, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Decision Tree)')

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
                        ('clf', DecisionTreeClassifier()),])  

    #Run_Classifier(0, 0, 1, pipeline, {}, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Decision Tree)')
    break  # Disable Cross Validation

Print_Result_Best()  # Best Cross Validation
###
quit()






Plot_Results("Movie Review Polarity Dataset")
quit()









### LET'S BUILD : SGDC-SVM
cross_validation_best = [0.000, "", [], []]
# Grid Search On
pipeline = Pipeline([
                    ('union', FeatureUnion(transformer_list=[      
                        ('vect1', CountVectorizer()),  # 1-Gram Vectorizer
                        ('vect2', CountVectorizer()),],  # 2-Gram Vectorizer
                    )),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, tol=None, n_jobs=-1)),]) 

parameters = {'clf__alpha': [1e-4, 1e-3, 1e-2],
              'tfidf__use_idf': [True],
              'union__transformer_weights': [{'vect1':1.0, 'vect2':1.0},],
              'union__vect1__max_df': [0.90, 0.80, 0.70],
              'union__vect1__min_df': [5, 8],  # 5 meaning 5 documents
              'union__vect1__ngram_range': [(1, 1)],              
              'union__vect1__stop_words': [stopwords.words("english"), 'english', stopwords_complete_lemmatized],
              'union__vect1__strip_accents': ['unicode'],
              'union__vect1__tokenizer': [LemmaTokenizer()],
              'union__vect2__max_df': [0.95, 0.85, 0.75],
              'union__vect2__min_df': [5, 8],
              'union__vect2__ngram_range': [(2, 2)],              
              'union__vect2__stop_words': [stopwords_complete_lemmatized, None],
              'union__vect2__strip_accents': ['unicode'],
              'union__vect2__tokenizer': [LemmaTokenizer()],} 

#Run_Classifier(1, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized, '(SGDC-SVM)')

# Grid Search Off
pipeline = Pipeline([ # Optimal
                    ('union', FeatureUnion(transformer_list=[      
                        ('vect1', CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                        ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                        transformer_weights={
                            'vect1': 1.0,
                            'vect2': 1.0,},
                    )),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                    ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold='2.5*mean')),  # Dimensionality Reduction 
                    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=1000, tol=None, n_jobs=-1)),]) 

Run_Classifier(0, 0, pipeline, {}, data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized, '(SGDC-SVM)')
###


### LET'S BUILD : SVM

# Grid Search On
pipeline = Pipeline([
                    ('union', FeatureUnion(transformer_list=[      
                        ('vect1', CountVectorizer()),  # 1-Grams Vectorizer
                        ('vect2', CountVectorizer()),],  # 2-Grams Vectorizer
                    )),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LinearSVC(loss='hinge', penalty='l2', max_iter=1000, dual=True)),])  # dual: True for Text/High Feature Count

parameters = {'clf__C': [1, 500, 1000],
              'tfidf__use_idf': [True],
              'union__transformer_weights': [{'vect1':1.0, 'vect2':1.0},],
              'union__vect1__max_df': [0.90, 0.80, 0.70],
              'union__vect1__min_df': [5, 8],  # 5 meaning 5 documents
              'union__vect1__ngram_range': [(1, 1)],              
              'union__vect1__stop_words': [stopwords.words("english"), 'english', stopwords_complete_lemmatized],
              'union__vect1__strip_accents': ['unicode'],
              'union__vect1__tokenizer': [LemmaTokenizer()],
              'union__vect2__max_df': [0.95, 0.85, 0.75],
              'union__vect2__min_df': [5, 8],
              'union__vect2__ngram_range': [(2, 2)],              
              'union__vect2__stop_words': [stopwords_complete_lemmatized, None],
              'union__vect2__strip_accents': ['unicode'],
              'union__vect2__tokenizer': [LemmaTokenizer()],} 

#Run_Classifier(1, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized, '(SVM)')

# Grid Search Off
pipeline = Pipeline([ # Optimal
                    ('union', FeatureUnion(transformer_list=[      
                        ('vect1', CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                        ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                        transformer_weights={
                            'vect1': 1.0,
                            'vect2': 1.0,},
                    )),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                    ('clf', LinearSVC(loss='hinge', penalty='l2', max_iter=1000, C=500, dual=True)),])  # dual: True for Text/High Feature Count

Run_Classifier(0, 0, pipeline, {}, data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized, '(SVM)')
###


### LET'S BUILD : Logistic Regression

# Grid Search On
pipeline = Pipeline([
                    ('union', FeatureUnion(transformer_list=[      
                        ('vect1', CountVectorizer()),  # 1-Grams Vectorizer
                        ('vect2', CountVectorizer()),],  # 2-Grams Vectorizer
                    )),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LogisticRegression(penalty='l2', max_iter=1000, dual=True)),])  # dual: True for Text/High Feature Count

parameters = {'clf__C': [1, 500, 1000],
              'tfidf__use_idf': [True],
              'union__transformer_weights': [{'vect1':1.0, 'vect2':1.0},],
              'union__vect1__max_df': [0.90, 0.80, 0.70],
              'union__vect1__min_df': [5, 8],  # 5 meaning 5 documents
              'union__vect1__ngram_range': [(1, 1)],              
              'union__vect1__stop_words': [stopwords.words("english"), 'english', stopwords_complete_lemmatized],
              'union__vect1__strip_accents': ['unicode'],
              'union__vect1__tokenizer': [LemmaTokenizer()],
              'union__vect2__max_df': [0.95, 0.85, 0.75],
              'union__vect2__min_df': [5, 8],
              'union__vect2__ngram_range': [(2, 2)],              
              'union__vect2__stop_words': [stopwords_complete_lemmatized, None],
              'union__vect2__strip_accents': ['unicode'],
              'union__vect2__tokenizer': [LemmaTokenizer()],} 

#Run_Classifier(1, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized, '(Logistic Regression)')

# Grid Search Off
pipeline = Pipeline([ # Optimal
                    ('union', FeatureUnion(transformer_list=[      
                        ('vect1', CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                        ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                        transformer_weights={
                            'vect1': 1.0,
                            'vect2': 1.0,},
                    )),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                    ('feature_selection', SelectKBest(score_func=chi2, k=5000)),  # Dimensionality Reduction
                    ('clf', LogisticRegression(penalty='l2', max_iter=1000, C=500, dual=True)),])  # dual: True for Text/High Feature Count

Run_Classifier(0, 0, pipeline, {}, data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized, '(Logistic Regression)')
###


### LET'S BUILD : MultiLayer Perceptron

# Grid Search On
pipeline = Pipeline([
                    ('union', FeatureUnion(transformer_list=[      
                        ('vect1', CountVectorizer()),  # 1-Grams Vectorizer
                        ('vect2', CountVectorizer()),],  # 2-Grams Vectorizer
                    )),
                    ('tfidf', TfidfTransformer()),
                    # Either of 2 Choices
                    #('feature_selection', SelectKBest(score_func=chi2, k=5000)),  # Dimensionality Reduction
                    ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold='2.5*mean')),  # Dimensionality Reduction  
                    ('clf', MLPClassifier(verbose=True)),])  

parameters = {'clf__hidden_layer_sizes': [(200,), (100,), (100,50,)],
              'clf__max_iter': [500],
              'clf__solver': ['sgd'],
              'clf__learning_rate': ['adaptive'],
              'clf__learning_rate_init': [0.60],
              'clf__momentum': [0.50],
              'clf__alpha': [1e-01, 1e-02, 1e-03, 1e-04, 1e-05, 1e-06],
              'tfidf__use_idf': [True],
              'union__transformer_weights': [{'vect1':1.0, 'vect2':1.0},],
              'union__vect1__max_df': [0.80,],
              'union__vect1__min_df': [5],  # 5 meaning 5 documents
              'union__vect1__ngram_range': [(1, 1)],              
              'union__vect1__stop_words': [stopwords_complete_lemmatized],
              'union__vect1__strip_accents': ['unicode'],
              'union__vect1__tokenizer': [LemmaTokenizer()],
              'union__vect2__max_df': [0.95],
              'union__vect2__min_df': [8],
              'union__vect2__ngram_range': [(2, 2)],              
              'union__vect2__stop_words': [None],
              'union__vect2__strip_accents': ['unicode'],
              'union__vect2__tokenizer': [LemmaTokenizer()],} 

#Run_Classifier(1, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized, '(MultiLayer Perceptron)')

# Grid Search Off
pipeline = Pipeline([ # Optimal
                    ('union', FeatureUnion(transformer_list=[      
                        ('vect1', CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                        ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                        transformer_weights={
                            'vect1': 1.0,
                            'vect2': 1.0,},
                    )),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                    # Either of 2 Choices
                    #('feature_selection', SelectKBest(score_func=chi2, k=5000)),  # Dimensionality Reduction
                    ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold='2.5*mean')),  # Dimensionality Reduction  
                    ('clf', MLPClassifier(verbose=False, hidden_layer_sizes=(200,), max_iter=500, solver='sgd', learning_rate='adaptive', learning_rate_init=0.60, momentum=0.50, alpha=1e-01)),])  

Run_Classifier(0, 0, pipeline, {}, data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized, '(MultiLayer Perceptron)')
###