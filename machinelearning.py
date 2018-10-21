'''
Sentiment Analysis Thesis: Text Classification using Naive Bayes,
'''

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.datasets import load_files
from sklearn.externals import joblib
from re import sub
import numpy as np
import string
import copy

def Run_Classifier(grid_search_enable, pickle_enable, pipeline, parameters, data_train, data_test, labels_train, labels_test, targetnames, stopwords_complete_lemmatized, model_name):
    '''    Run Classifier with or without Grid Search after Preprocessing is done    '''

    ## PREPARE ON - Grid Search to Look for the Best Parameters
    if grid_search_enable == 1:

        # (1) TRAIN
        grid_go = GridSearchCV(pipeline, parameters, n_jobs=-1)
        grid_go = grid_go.fit(data_train, labels_train)
        print('- - - - - BEST PARAMETERS - - - - -')
        print(grid_go.best_score_, 'Accuracy')
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, grid_go.best_params_[param_name]))

        print('\n- - - - - DETAILS - - - - -')
        for i in range(len(grid_go.cv_results_['params'])):
            results_noStopWords = copy.deepcopy(grid_go.cv_results_['params'][i])
            if model_name != '(MultiLayer Perceptron)':
                if results_noStopWords['union__vect1__stop_words'] is not None:  # Don't Print the list of Stopwords
                    results_noStopWords['union__vect1__stop_words'] = ['ListOfStopWords']   
                if results_noStopWords['union__vect2__stop_words'] is not None:
                    results_noStopWords['union__vect2__stop_words'] = ['ListOfStopWords']           
            print(i, 'params - %s; mean - %0.10f; std - %0.10f'
                        % (results_noStopWords.values(),
                        grid_go.cv_results_['mean_test_score'][i],
                        grid_go.cv_results_['std_test_score'][i]))

        # (2) Model Persistence (Pickle)
        if pickle_enable == 1: joblib.dump(grid_go.best_estimator_, './pickled_models/review_polarity/Classifier.pkl')  

        # (3) PREDICT
        predicted = grid_go.predict(data_test)

    ## PREPARE OFF - Best Parameters are already known
    else:   

        # (1) TRAIN
        pipeline.fit(data_train, labels_train)
        if model_name != '(MultiLayer Perceptron)': print('\nNumber of Features/Dimension is:', pipeline.named_steps['clf'].coef_.shape[1])

        # (2) Model Persistence (Pickle)
        if pickle_enable == 1: joblib.dump(pipeline, './pickled_models/review_polarity/Classifier.pkl') 

        # (3) PREDICT
        predicted = pipeline.predict(data_test)

    Print_Result_Metrics(labels_test, predicted, targetnames, model_name)  

def Print_Result_Metrics(labels_test, predicted, targetnames, model_name):
    '''    Print Metrics after Training etc.    '''
    print('\n- - - - - RESULT METRICS', model_name, '- - - - -')
    print('Exact Accuracy: ', metrics.accuracy_score(labels_test, predicted))
    print(metrics.classification_report(labels_test, predicted, target_names=targetnames))
    print(metrics.confusion_matrix(labels_test, predicted))

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


### PREPROCESSING
dataset = load_files('./datasets/review_polarity/txt_sentoken', shuffle=False)

stopwords_complete = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
wnl = WordNetLemmatizer()
stopwords_complete_lemmatized = set([wnl.lemmatize(word) for word in stopwords_complete])

np.set_printoptions(precision=10)  # Numpy Precision when Printing

# Split, data & labels are pairs
data_train, data_test, labels_train, labels_test = train_test_split(dataset.data, dataset.target, test_size=0.30, random_state=22)

# Dimensionality Reduction - 4 different ways to pick the best Features 
#   (1) ('feature_selection', SelectKBest(score_func=chi2, k=5000)),                    
#   (2) ('feature_selection', TruncatedSVD(n_components=1000)),  # Has Many Issues
#   (3) ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold='2.5*mean')),
#   (4) ('feature_selection', SelectFromModel(estimator=LinearSVC(penalty='l1', dual=False), threshold='mean')),  # Technically L1 is better than L2
#   Variance VarianceThreshold


### LET'S BUILD : Naive Bayes

# Grid Search On
pipeline = Pipeline([
                    ('union', FeatureUnion(transformer_list=[      
                        ('vect1', CountVectorizer()),  # 1-Grams Vectorizer
                        ('vect2', CountVectorizer()),],  # 2-Grams Vectorizer
                    )),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),])  

parameters = {'tfidf__use_idf': [True],
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

#Run_Classifier(1, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized, '(Naive Bayes)')

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
                    ('clf', MultinomialNB()),])  

Run_Classifier(0, 0, pipeline, {}, data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized, '(Naive Bayes)')
###


### LET'S BUILD : SGDC-SVM

# Grid Search On
pipeline = Pipeline([
                    ('union', FeatureUnion(transformer_list=[      
                        ('vect1', CountVectorizer()),  # 1-Grams Vectorizer
                        ('vect2', CountVectorizer()),],  # 2-Grams Vectorizer
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