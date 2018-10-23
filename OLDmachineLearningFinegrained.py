''' Naive Bayes on the Finegrained Dataset '''

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

import pandas as pd
from pomegranate import *

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

dataset = [["" for j in range(3)] for i in range(294)]
count = 0
with open('finegrained.txt', 'r') as file:
    for line in file:
        if len(line.split("_")) == 3:
            dataset[count][0] = line.split("_")[1]
        elif len(line.strip()) == 0:
            count += 1
        else:
            temp = [x.strip() for x in line.split("\t")]
            if len(temp[1]) > 1:
                # n for negative - u for neutral - p for positive - m for mix --- nr is ignored
                if temp[0].startswith("neg"):
                    dataset[count][1] += "n"
                elif temp[0].startswith("neu"):
                    dataset[count][1] += "u"
                elif temp[0].startswith("pos"):
                    dataset[count][1] += "p"
                elif temp[0].startswith("mix"):
                    dataset[count][1] += "m"                    

                dataset[count][2] += temp[1]


print("--Processed", count+1, "documents")

df_dataset = pd.DataFrame(data=dataset)

# Split, data and train
df_train, df_test = train_test_split(df_dataset, test_size=0.20, random_state=22)
data_train = df_train.iloc[:,1]
labels_train = df_train.iloc[:,0]
data_test = df_test.iloc[:,1]
labels_test = df_test.iloc[:,0]

print("\nDataset Info:\n", df_dataset.describe(include="all"), "\n", df_dataset.head(), "\n")


stopwords_complete = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
wnl = WordNetLemmatizer()
stopwords_complete_lemmatized = set([wnl.lemmatize(word) for word in stopwords_complete])

np.set_printoptions(precision=10)  # Numpy Precision when Printing

# The sequences are stored as a String in the Dataframe, time to transform them to the correct form
data_train_asList = [[0 for j in range(4)] for i in range(len(data_train))]
for i, x in enumerate(data_train):
    tempSeq = list(x)
    countNeg = 0  
    countNeu = 0     
    countPos = 0
    countMix = 0
    for y in tempSeq:
        if y == 'n':
            countNeg += 1
        elif y == 'u':
            countNeu += 1
        elif y == 'p':
            countPos += 1
        elif y == 'm':
            countMix += 1                    
    data_train_asList[i][0] = countNeg
    data_train_asList[i][1] = countNeu
    data_train_asList[i][2] = countPos
    data_train_asList[i][3] = countMix
# Same
data_test_asList = [[0 for j in range(4)] for i in range(len(data_test))]
for i, x in enumerate(data_test):
    tempSeq = list(x)
    countNeg = 0  
    countNeu = 0     
    countPos = 0
    countMix = 0
    for y in tempSeq:
        if y == 'n':
            countNeg += 1
        elif y == 'u':
            countNeu += 1
        elif y == 'p':
            countPos += 1
        elif y == 'm':
            countMix += 1                    
    data_test_asList[i][0] = countNeg
    data_test_asList[i][1] = countNeu
    data_test_asList[i][2] = countPos
    data_test_asList[i][3] = countMix


### LET'S BUILD : Naive Bayes

# Grid Search Off
pipeline = Pipeline([ # Optimal
                    #('union', FeatureUnion(transformer_list=[      
                    #    ('vect1', CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                    #    ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                    #    transformer_weights={
                    #        'vect1': 1.0,
                    #        'vect2': 1.0,},
                    #)),
                    #('tfidf', TfidfTransformer(use_idf=True)),    
                    #('feature_selection', SelectKBest(score_func=chi2, k=5000)),  # Dimensionality Reduction                   
                    ('clf', MultinomialNB()),])  

Run_Classifier(0, 0, pipeline, {}, data_train_asList, data_test_asList, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Naive Bayes)')
###


### LET'S BUILD : SVM

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
                    ('clf', LinearSVC(loss='hinge', penalty='l2', max_iter=1000, C=500)),])  # dual: True for Text/High Feature Count

#Run_Classifier(0, 0, pipeline, {}, data_train_asList, data_test_asList, labels_train, labels_test, None, stopwords_complete_lemmatized, '(SVM)')
###


### LET'S BUILD : Logistic Regression

# Grid Search Off
pipeline = Pipeline([ # Optimal
                    #('union', FeatureUnion(transformer_list=[      
                    #    ('vect1', CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                    #    ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                    #    transformer_weights={
                    #        'vect1': 1.0,
                    #        'vect2': 1.0,},
                    #)),
                    #('tfidf', TfidfTransformer(use_idf=True)),
                    #('feature_selection', SelectKBest(score_func=chi2, k=5000)),  # Dimensionality Reduction
                    ('clf', LogisticRegression(penalty='l2', max_iter=1000, C=500)),])  # dual: True for Text/High Feature Count

Run_Classifier(0, 0, pipeline, {}, data_train_asList, data_test_asList, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Logistic Regression)')
###


### LET'S BUILD : MultiLayer Perceptron

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

#Run_Classifier(0, 0, pipeline, {}, data_train_asList, data_test_asList, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized, '(MultiLayer Perceptron)')
###