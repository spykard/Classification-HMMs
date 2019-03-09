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
    # data = [["" for j in range(3)] for i in range(294)]
    data = ["" for i in range(294)]
    neg_count_feature = [0 for i in range(294)]  # 3 Extra columns, 3 Extra Features containing the counts (no sequential information)
    neu_count_feature = [0 for i in range(294)]
    pos_count_feature = [0 for i in range(294)]
    labels = ["" for i in range(294)]
    count = 0
    with open('./Datasets/Finegrained/finegrained.txt', 'r') as file:
        for line in file:
            if len(line.split("_")) == 3:
                labels[count] = line.split("_")[1]
            elif len(line.strip()) == 0:
                count += 1
            else:
                temp = [x.strip() for x in line.split("\t")]
                if len(temp[1]) > 1:
                    # "nr" label is ignored
                    if temp[0] == "neg":
                        neg_count_feature[count] += 1
                    elif temp[0] == "neu":
                        neu_count_feature[count] += 1
                    elif temp[0] == "pos":
                        pos_count_feature[count] += 1
                    elif temp[0] == "mix":
                        neg_count_feature[count] += 1
                        pos_count_feature[count] += 1        

                    data[count] += temp[1]

    print("--Processed", count+1, "documents", "\n--Dataset Name:", dataset_name)

    df_dataset = pd.DataFrame({'Labels': labels, 'Data': data, 'Neg_Feature': neg_count_feature, 'Neu_Feature': neu_count_feature, 'Pos_Feature': pos_count_feature})

    # 2. Remove empty instances from DataFrame, actually affects accuracy
    emptyCells = df_dataset.loc[df_dataset.loc[:,'Data'].map(len) < 1].index.values
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
        model_names.append(model[1:-1])  # Remove Parentheses

    # Reverse the items to appear in correct order
    scores_acc.reverse()
    scores_f1.reverse()
    model_names.reverse()

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

df_dataset = Run_Preprocessing("Finegrained Sentiment Dataset")
all_data = df_dataset.loc[:,'Data']
all_labels = df_dataset.loc[:,'Labels']
neg_feature = df_dataset.loc[:,'Neg_Feature']
neu_feature = df_dataset.loc[:,'Neu_Feature']
pos_feature = df_dataset.loc[:,'Pos_Feature']

global_dataframe = df_dataset

print("\n--Dataset Info:\n", df_dataset.describe(include="all"), "\n\n", df_dataset.head(), "\n\n", df_dataset.loc[:,'Labels'].value_counts(), "\n--\n", sep="")

# Split using Cross Validation
set_fold = 5
cross_validation_enable = True  # Enable/Disable Flag; if disabled runs the evaluation just once
k_fold = RepeatedStratifiedKFold(5, n_repeats=1, random_state=22)


# Dimensionality Reduction - 4 different ways to pick the best Features 
#   (1) ('feature_selection', SelectKBest(score_func=chi2, k=5000)),  # 0.852 accuracy                    
#   (2) ('feature_selection', TruncatedSVD(n_components=1000)),  # Has Many Issues
#   (3) ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold=-np.inf, max_features=5000)),  # 0.860 accuracy 
#   (4) ('feature_selection', SelectFromModel(estimator=LinearSVC(penalty='l1', dual=False), threshold=-np.inf, max_features=5000)),  # 0.824 accuracy | Technically L1 is better than L2

average = []

### (1) LET'S BUILD : Complement Naive Bayes
cross_validation_best = [0.0, 0.0, "", [], [], 0.0]  # [Accuracy, F1-score, Model Name, Actual Labels, Predicted, Time]
for k, (train_indexes, test_indexes) in enumerate(k_fold.split(all_data, all_labels)):  # Split must be done before every classifier because generated object gets exhausted (destroyed)
    print("\n--Current Cross Validation Fold:", k+1)

    data_train = all_data.reindex(train_indexes, copy=True, axis=0)
    labels_train = all_labels.reindex(train_indexes, copy=True, axis=0)
    neg_feature_train = neg_feature.reindex(train_indexes, copy=True, axis=0)
    neu_feature_train = neu_feature.reindex(train_indexes, copy=True, axis=0)
    pos_feature_train = pos_feature.reindex(train_indexes, copy=True, axis=0)

    data_test = all_data.reindex(test_indexes, copy=True, axis=0)
    labels_test = all_labels.reindex(test_indexes, copy=True, axis=0)
    neg_feature_test = neg_feature.reindex(test_indexes, copy=True, axis=0)
    neu_feature_test = neu_feature.reindex(test_indexes, copy=True, axis=0)
    pos_feature_test = pos_feature.reindex(test_indexes, copy=True, axis=0)


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
                        #('feature_selection', SelectKBest(score_func=chi2, k=5000)),  # Dimensionality Reduction                  
                        ('clf', ComplementNB()),])  

    #Run_Classifier(0, 0, 1, pipeline, {}, data_train, data_test, labels_train, labels_test, None, stopwords_complete_lemmatized, '(Complement Naive Bayes)')

    # Training    
    vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized)
    tfidf_matrix = vectorizer.fit_transform(data_train)
    tfidf_dense = tfidf_matrix.toarray()  # dense ndarray
    print(type(tfidf_dense), tfidf_dense.shape)
    print(tfidf_dense)

    #b = np.array([[100, 200], [100, 200], [100, 200], [100, 200], [100, 200], [100, 200]])
    #b = b[:, None]
    #print("The shapes are:",tfidf_dense.shape, b.shape) 
    #success = np.column_stack((tfidf_dense, b))  # or hstack
    neg_feature_train = neg_feature_train.values[:, None]
    neu_feature_train = neu_feature_train.values[:, None]    
    pos_feature_train = pos_feature_train.values[:, None]  
    print(tfidf_dense.shape)
    tfidf_dense = np.hstack((tfidf_dense, neg_feature_train, neu_feature_train, pos_feature_train))  # or column_stack   
    print(tfidf_dense.shape)
    #quit()

    clf = ComplementNB()
    clf.fit(tfidf_dense, labels_train)

    # Test
    tf_idf_matrix_test = vectorizer.transform(data_test)
    tfidf_dense_test = tf_idf_matrix_test.toarray()  # dense ndarray

    neg_feature_test = neg_feature_test.values[:, None]
    neu_feature_test = neu_feature_test.values[:, None]    
    pos_feature_test = pos_feature_test.values[:, None]  
    print(tfidf_dense_test.shape)
    tfidf_dense_test = np.hstack((tfidf_dense_test, neg_feature_test, neu_feature_test, pos_feature_test))  # or column_stack   
    print(tfidf_dense_test.shape)


    predicted = clf.predict(tfidf_dense_test)


    accuracy = metrics.accuracy_score(labels_test, predicted)
    other_metrics_to_print = metrics.classification_report(labels_test, predicted, output_dict=False)
    other_metrics_as_dict = metrics.classification_report(labels_test, predicted, output_dict=True)
    confusion_matrix = metrics.confusion_matrix(labels_test, predicted)

    print('\n- - - - - RESULT METRICS - - - - - -')
    print('Exact Accuracy: ', accuracy)
    print(other_metrics_to_print)
    print(confusion_matrix)
    print()

    average.append(other_metrics_as_dict['weighted avg']['f1-score'])


    #Print_Result_Metrics(0, labels_test, predicted, None, 0.1, 0, "ComplementNB")  

    if cross_validation_enable == False:
        break  # Disable Cross Validation

Print_Result_CrossVal_Best(k)
print(np.mean(average))

###
quit()




Print_Result_CrossVal_Best(k)
###

# Don't print, just plot the averages of all models
Plot_Results(set_fold, "Finegrained Sentiment Dataset")