# DOES NOT WORK CORRECTLY, ngram_3 gives lower results than it normally would
import pickle
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
# Ensemble
# Pickle Save
# with open('./Pickled Objects/3-gram-predictions', 'wb') as f:
#     pickle.dump(hmm.cross_val_prediction_matrix, f)
# Pickle Load
k_fold = 5
random_state = 22
decimals = 3
cross_val = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=1, random_state=random_state)
mapping = {0: "neg", 1: "neu", 2: "pos"}
observations = df.loc[:,"Sequences"].values
golden_truth = df.loc[:,"Labels"].values
f1_scores = []
acc_scores = []

ngram_1 = pickle.load(open('./Pickled Objects/1-gram-predictions', 'rb'))
ngram_2 = pickle.load(open('./Pickled Objects/2-gram-predictions', 'rb'))
ngram_3 = pickle.load(open('./Pickled Objects/3-gram-predictions', 'rb'))

for i, (train_index, test_index) in enumerate(cross_val.split(observations, golden_truth)):
    ensemble = 0.5*ngram_1[i] + 0.1*ngram_2[i] + 0.0*ngram_3[i]  # Note that there are problems with -ninf values
    indices = np.argmax(ensemble, axis=1)
    prediction = []
    for x in indices:
        prediction.append(mapping[x])

    # Metrics
    accuracy = metrics.accuracy_score(golden_truth[test_index], prediction)
    rest_as_string = metrics.classification_report(golden_truth[test_index], prediction, output_dict=False)  # Used as a string
    rest_as_dict = metrics.classification_report(golden_truth[test_index], prediction, output_dict=True)  # Used as an information source
    confusion_matrix = metrics.confusion_matrix(golden_truth[test_index], prediction)     

    f1_scores.append(rest_as_dict['weighted avg']['f1-score'])
    acc_scores.append(accuracy)

print(np.around(np.mean(f1_scores)*100.0, decimals=decimals))
print(np.around(np.mean(acc_scores)*100.0, decimals=decimals))