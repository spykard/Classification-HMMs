""" 
Sentiment Analysis: (mainly Supervised) Text Classification using Hidden Markov Models
"""

import pandas as pd
import numpy as np
import HMM_Framework


dataset_name = "IMDb Large Movie Review Dataset"
random_state = 22

# 1. Dataset dependent loading - Load artificial labels
import pickle

artifically_labeled_data = pickle.load(open('./Pickled Objects/Artificial_Labels_2/Artificial_Labels_from_Bayes', 'rb'))
data_corresponding_to_labels = pickle.load(open('./Pickled Objects/Artificial_Labels_2/Data_corresponding_to_Labels_from_Bayes', 'rb'))
golden_truth = pickle.load(open('./Pickled Objects/Artificial_Labels_2/Artifical_Labels_Golden_Truth', 'rb'))

artifically_labeled_data_test = pickle.load(open('./Pickled Objects/Artificial_Labels_2/Artificial_Labels_from_Bayes_Test_Set', 'rb'))
data_corresponding_to_labels_test = pickle.load(open('./Pickled Objects/Artificial_Labels_2/Data_corresponding_to_Labels_from_Bayes_Test_Set', 'rb'))
golden_truth_test = pickle.load(open('./Pickled Objects/Artificial_Labels_2/Artifical_Labels_Golden_Truth_Test_Set', 'rb'))

artifically_labeled_data_merged = artifically_labeled_data + artifically_labeled_data_test
data_corresponding_to_labels_merged = data_corresponding_to_labels + data_corresponding_to_labels_test
golden_truth_merged = golden_truth + golden_truth_test

print("--\n--Processed", len(golden_truth_merged), "documents", "\n--Dataset Name:", dataset_name)

df = pd.DataFrame({'Labels': golden_truth_merged, 'Words': data_corresponding_to_labels_merged, 'Artificial_Labels': artifically_labeled_data_merged})

# 2. Remove empty instances from DataFrame, actually affects accuracy
emptySequences = df.loc[df.loc[:,'Artificial_Labels'].map(len) < 1].index.values
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


# MAIN
# AdvancedHMM.build
#       General Settings
#       Data
#       Text Scenario
#       n-gram Settings
#       1st Framework Training Settings (High-Order done through the n-grams Settings)
#       1st Framework Prediction Settings (Achitecture A)
#       2nd Framework Training Settings (High-Order done through the 'hohmm_high_order' parameter)
#       Any Framework Prediction Settings (Architecture B)

if True:
    #  Model
    #  Just for State-emission HMM, might need to remove the "mix" label during preprocessing.
    hmm = HMM_Framework.HMM_Framework()
    hmm.build(architecture="B", model="Classic HMM", framework="hohmm", k_fold=5,                                                   \
            state_labels_pandas=df.loc[:,"Artificial_Labels"], observations_pandas=df.loc[:,"Words"], golden_truth_pandas=df.loc[:,"Labels"], \
            text_instead_of_sequences=[], text_enable=False,                                                                              \
            n_grams=1, n_target="both", n_prev_flag=False, n_dummy_flag=False,                                                           \
            pome_algorithm="baum-welch", pome_verbose=False, pome_njobs=1, pome_smoothing_trans=0.0, pome_smoothing_obs=0,                \
            pome_algorithm_t="map",                                                                                                       \
            hohmm_high_order=1, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
            architecture_b_algorithm="formula", formula_magic_smoothing=0                                                             \
            )   

hmm.print_average_results(decimals=3)
hmm.print_best_results(detailed=True, decimals=3) 