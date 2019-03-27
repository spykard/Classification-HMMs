""" 
Sentiment Analysis: Text Classification using Hidden Markov Models
"""

import pandas as pd
import numpy as np
import HMM_Framework
import Ensemble_Framework


dataset_name = "Finegrained Sentiment Dataset"
random_state = 22

# 1. Dataset dependent loading
data = ["" for i in range(294)]
sequences = [[] for i in range(294)]
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
                if temp[0] in ["neg", "neu", "pos"]:
                    sequences[count].append(temp[0])              

                data[count] += temp[1]

print("--\n--Processed", count+1, "documents", "\n--Dataset Name:", dataset_name)

df = pd.DataFrame({'Labels': labels, 'Data': data, 'Sequences': sequences})

# 2. Remove empty instances from DataFrame, actually affects accuracy
emptySequences = df.loc[df.loc[:,'Sequences'].map(len) < 1].index.values
df = df.drop(emptySequences, axis=0).reset_index(drop=True)  # reset_Index to make the row numbers be consecutive again

# 3. Shuffle the Dataset, just to make sure it's not too perfectly ordered
if False:
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

elif False:
    # n-gram Ensemble
    cross_val_prediction_matrix = []
    mapping = []
    golden_truth = []

    # Make sure that Flags are the exact same on all 3
    general_mixture_model_labels = HMM_Framework.general_mixture_model_label_generator(df.loc[:,"Sequences"], df.loc[:,"Labels"])
    hmm = HMM_Framework.HMM_Framework()
    hmm.build(architecture="A", model="General Mixture Model", framework="pome", k_fold=5,                                                \
            state_labels_pandas=general_mixture_model_labels, observations_pandas=df.loc[:,"Sequences"], golden_truth_pandas=df.loc[:,"Labels"], \
            text_instead_of_sequences=[], text_enable=False,                                                                              \
            n_grams=1, n_target="obs", n_prev_flag=False, n_dummy_flag=False,                                                             \
            pome_algorithm="baum-welch", pome_verbose=False, pome_njobs=1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
            pome_algorithm_t="map",                                                                                                       \
            hohmm_high_order=1, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
            architecture_b_algorithm="forward", formula_magic_smoothing=0.0                                                               \
            )   

    cross_val_prediction_matrix.append(hmm.cross_val_prediction_matrix)
    mapping.append(hmm.ensemble_stored["Mapping"])
    golden_truth.append(hmm.ensemble_stored["Curr_Cross_Val_Golden_Truth"])

    hmm = HMM_Framework.HMM_Framework()
    hmm.build(architecture="A", model="General Mixture Model", framework="pome", k_fold=5,                                                \
            state_labels_pandas=general_mixture_model_labels, observations_pandas=df.loc[:,"Sequences"], golden_truth_pandas=df.loc[:,"Labels"], \
            text_instead_of_sequences=[], text_enable=False,                                                                              \
            n_grams=2, n_target="obs", n_prev_flag=False, n_dummy_flag=False,                                                             \
            pome_algorithm="baum-welch", pome_verbose=False, pome_njobs=1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
            pome_algorithm_t="map",                                                                                                       \
            hohmm_high_order=1, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
            architecture_b_algorithm="forward", formula_magic_smoothing=0.0                                                               \
            )   

    cross_val_prediction_matrix.append(hmm.cross_val_prediction_matrix)
    mapping.append(hmm.ensemble_stored["Mapping"])
    golden_truth.append(hmm.ensemble_stored["Curr_Cross_Val_Golden_Truth"])

    hmm = HMM_Framework.HMM_Framework()
    hmm.build(architecture="A", model="General Mixture Model", framework="pome", k_fold=5,                                                \
            state_labels_pandas=general_mixture_model_labels, observations_pandas=df.loc[:,"Sequences"], golden_truth_pandas=df.loc[:,"Labels"], \
            text_instead_of_sequences=[], text_enable=False,                                                                              \
            n_grams=3, n_target="obs", n_prev_flag=False, n_dummy_flag=False,                                                             \
            pome_algorithm="baum-welch", pome_verbose=False, pome_njobs=1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
            pome_algorithm_t="map",                                                                                                       \
            hohmm_high_order=1, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
            architecture_b_algorithm="forward", formula_magic_smoothing=0.0                                                               \
            )   

    cross_val_prediction_matrix.append(hmm.cross_val_prediction_matrix)
    mapping.append(hmm.ensemble_stored["Mapping"])
    golden_truth.append(hmm.ensemble_stored["Curr_Cross_Val_Golden_Truth"])

    Ensemble_Framework.ensemble_run(cross_val_prediction_matrix, mapping, golden_truth, mode="sum", weights=[0.20, 0.40, 0.40])

elif True:
    # high-order Ensemble
    cross_val_prediction_matrix = []
    mapping = []
    golden_truth = []

    # Make sure that Settings, other than the order, are the exact same on all 3
    hmm = HMM_Framework.HMM_Framework()
    hmm.build(architecture="B", model="State-emission HMM", framework="pome", k_fold=5, boosting=False,                                   \
            state_labels_pandas=df.loc[:,"Sequences"], observations_pandas=df.loc[:,"Sequences"], golden_truth_pandas=df.loc[:,"Labels"], \
            text_instead_of_sequences=[], text_enable=False,                                                                              \
            n_grams=1, n_target="both", n_prev_flag=False, n_dummy_flag=False,                                                             \
            pome_algorithm="baum-welch", pome_verbose=False, pome_njobs=1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
            pome_algorithm_t="map",                                                                                                       \
            hohmm_high_order=1, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
            architecture_b_algorithm="formula", formula_magic_smoothing=0.0                                                               \
            )  

    cross_val_prediction_matrix.append(hmm.cross_val_prediction_matrix)
    mapping.append(hmm.ensemble_stored["Mapping"])
    golden_truth.append(hmm.ensemble_stored["Curr_Cross_Val_Golden_Truth"])

    hmm = HMM_Framework.HMM_Framework()
    hmm.build(architecture="B", model="State-emission HMM", framework="pome", k_fold=5, boosting=False,                                   \
            state_labels_pandas=df.loc[:,"Sequences"], observations_pandas=df.loc[:,"Sequences"], golden_truth_pandas=df.loc[:,"Labels"], \
            text_instead_of_sequences=[], text_enable=False,                                                                              \
            n_grams=2, n_target="both", n_prev_flag=False, n_dummy_flag=False,                                                             \
            pome_algorithm="baum-welch", pome_verbose=False, pome_njobs=1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
            pome_algorithm_t="map",                                                                                                       \
            hohmm_high_order=2, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
            architecture_b_algorithm="formula", formula_magic_smoothing=0.0                                                               \
            )   

    cross_val_prediction_matrix.append(hmm.cross_val_prediction_matrix)
    mapping.append(hmm.ensemble_stored["Mapping"])
    golden_truth.append(hmm.ensemble_stored["Curr_Cross_Val_Golden_Truth"])

    hmm = HMM_Framework.HMM_Framework()
    hmm.build(architecture="B", model="State-emission HMM", framework="pome", k_fold=5, boosting=False,                                   \
            state_labels_pandas=df.loc[:,"Sequences"], observations_pandas=df.loc[:,"Sequences"], golden_truth_pandas=df.loc[:,"Labels"], \
            text_instead_of_sequences=[], text_enable=False,                                                                              \
            n_grams=2, n_target="states", n_prev_flag=False, n_dummy_flag=True,                                                             \
            pome_algorithm="baum-welch", pome_verbose=False, pome_njobs=1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,              \
            pome_algorithm_t="map",                                                                                                       \
            hohmm_high_order=1, hohmm_smoothing=0.0, hohmm_synthesize=False,                                                              \
            architecture_b_algorithm="formula", formula_magic_smoothing=0.0                                                               \
            )     

    cross_val_prediction_matrix.append(hmm.cross_val_prediction_matrix)
    mapping.append(hmm.ensemble_stored["Mapping"]) 
    golden_truth.append(hmm.ensemble_stored["Curr_Cross_Val_Golden_Truth"])

    Ensemble_Framework.ensemble_run(cross_val_prediction_matrix, mapping, golden_truth, mode="sum", weights=[0.4, 0.3, 0.3])
