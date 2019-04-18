""" 
HMM-Framework: An advanced framework that implements state-of-the-art Hidden Markov Models, mostly for supervised/classification tasks.
"""

import numpy as np
import itertools
from collections import defaultdict
import pandas as pd
import pomegranate as pome  # Import it in this manner to avoid conflicts with the 'time' module
import SimpleHOHMM
import time
import random
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from nltk import ngrams as ngramsgenerator
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn import metrics
import Ensemble_Framework


random_state = 22
random.seed(22)

class HMM_Framework:
    """
        observations: Observation sequence
        state_labels: Hidden State sequence
        A: Transition probability matrix
        B: Observation probability matrix
        pi: Initial probabilities
    """
    def __init__(self):
        self.state_labels = []
        self.observations = []
        self.golden_truth = []
        self.A = []
        self.B = []
        self.pi = []

        self.text_data = []
        self.text_enable = False        
        self.length = 0
        self.architectures = ["A", "B"]
        self.selected_architecture = ""
        self.models = ["General Mixture Model", "State-emission HMM", "Classic HMM"]
        self.selected_model = ""
        self.frameworks = ["pome", "hohmm"]
        self.selected_framework = ""
        self.multivariate = False

        self.trained_model = []  # The object that is outputed after training on the current cross validation fold; depends on the framework that was used
        self.unique_states = set()
        self.unique_states_subset = set()
        self.unique_observations_subset = set()
        self.state_to_label_mapping = []        # Dict: {"pos": 0, "neg": 1, ...}
        self.state_to_label_mapping_rev = []    # Dict: {0: "pos", 1: "neg", ...}
        self.observation_to_label_mapping = []  # Dict: {"good": 0, ambitious: 1, ...}
        self.hmm_to_label_mapping = {}          # Used for Architecture 'B', {Model_0: "pos", Model_1: "neg", ...}

        # Evaluation
        self.k_fold = 0
        self.cross_val_metrics = defaultdict(list)  # {Name: [], 
                                                    # F1-score: [], 
                                                    # Accuracy: [], 
                                                    # Metrics_String: [], 
                                                    # Confusion_Matrix: [],
                                                    # Time_Complexity: []}
        self.cross_val_prediction_matrix = []       # The final predictions for all the folds of cross validation
        self.count_new_oov = []                     # Count the number of instances where we encountered out-of-vocabulary new observations
        self.count_formula_problems = []            # Count the number of instances where the 'formula' algorithm prediction failed and was forced to perform random guessing

        # Stored for future Ensembles
        self.ensemble_stored = defaultdict(list)  # {Mapping: [], 
                                                  # Curr_Cross_Val_Golden_Truth: []}

    def clean_up(self):
        """
        Resets the values which are not needed to their original values, after the entire process is finished.
        """
        self.state_labels = []
        self.observations = []
        self.text_data = []        

    def reset(self):
        """
        Important. Resets some important variables after each cross validation fold.
        """  
        self.trained_model = []
        self.unique_states_subset = set()
        self.state_to_label_mapping = []
        self.state_to_label_mapping_rev = []
        self.observation_to_label_mapping = []
        self.hmm_to_label_mapping = {}

    def check_input_type(self, state_labels_pandas, observations_pandas, golden_truth_pandas, text_instead_of_sequences, text_enable):
        """
        Make sure that the input data are of the correct type: pandas Series. Also perform the assignment to the local variables.
        """
        self.text_enable = text_enable
        if self.text_enable == False:
            if (isinstance(state_labels_pandas, pd.Series) != True) or (isinstance(observations_pandas, pd.Series) != True):
                raise ValueError("please make sure that you are inputting the parameters 'state_labels_pandas' and 'observations_pandas' in the form of a pandas Series, i.e. select a column of a DataFrame.")
            else:
                self.state_labels = copy.deepcopy(state_labels_pandas.values)  # or tolist() to get list instead of ndarray; Should be Deep Copy in case the same array is given on both inputs, be careful
                self.observations = copy.deepcopy(observations_pandas.values)
        else:
            if (isinstance(text_instead_of_sequences, pd.Series) != True):  # just text parameter
                raise ValueError("please make sure that you are inputting the parameter 'text_instead_of_sequences' in the form of a pandas Series, i.e. select a column of a DataFrame.")
            else:
                self.text_data = copy.deepcopy(text_instead_of_sequences.values)         
            if len(state_labels_pandas) > 0:  # text + state_labels parameters
                if (isinstance(state_labels_pandas, pd.Series) != True):
                    raise ValueError("text parameter was given correctly but please make sure that you are inputting the parameters 'state_labels_pandas' in the form of a pandas Series, i.e. select a column of a DataFrame.")
                else:
                    self.state_labels = copy.deepcopy(state_labels_pandas.values)
            if len(observations_pandas) > 0:  # text + observations parameters
                if (isinstance(observations_pandas, pd.Series) != True):
                    raise ValueError("text parameter was given correctly but please make sure that you are inputting the parameter 'observations_pandas' in the form of a pandas Series, i.e. select a column of a DataFrame.")               
                else:
                    self.observations = copy.deepcopy(observations_pandas.values)

        if (isinstance(golden_truth_pandas, pd.Series) != True):
            raise ValueError("please make sure that you are inputting the parameter 'golden_truth_pandas' in the form of a pandas Series, i.e. select a column of a DataFrame.")
        else:           
            self.golden_truth = copy.deepcopy(golden_truth_pandas.values) 

    def verify_and_autodetect(self):
        """
        (1) Ensure that the input data are of the same shape.
        (2) Automatically detect some basic HMMs that could be useful and suggest them to the user.
        (3) Print the approach that the user has selected.
        """
        # Verify shape validity
        if self.text_enable == False:        
            if len(self.state_labels) == 0 or len(self.observations) == 0:
                raise ValueError("one or both of the input containers appear to be empty.")
            elif len(self.state_labels) != len(self.observations):
                raise ValueError("the first input container is of length " + str(len(self.state_labels)) + " while the second is of length " + str(len(self.observations)) + ".")
            else:
                for i in range(len(self.state_labels)):
                    if len(self.state_labels[i]) != len(self.observations[i]):
                        raise ValueError("on row with index " + str(i) + ", the sequence of the first container is of size " + str(len(self.state_labels[i])) + " while the one of the second is of size " + str(len(self.observations[i])) + ".")
        else:
            if len(self.text_data) == 0:
                raise ValueError("the text input container appears to be empty.")
            if len(self.state_labels) > 0:
                if len(self.text_data) != len(self.state_labels):
                    raise ValueError("you want to use the first input container but it is of length " + str(len(self.state_labels)) + " while the text one is of length " + str(len(self.text_data)) + ".")                 
            if len(self.observations) > 0:
                if len(self.text_data) != len(self.observations):
                    raise ValueError("you want to use the second input container but it is of length " + str(len(self.observations)) + " while the text one is of length " + str(len(self.text_data)) + ".")         

        if len(self.golden_truth) != len(self.observations):
            raise ValueError("the golden truth list is of length " + str(len(self.golden_truth)) + " while the observation container is of length " + str(len(self.observations)) + ".")    

        if self.selected_architecture not in self.architectures:
            raise ValueError("selected architecture does not exist.")
        if self.selected_model not in self.models:
            raise ValueError("selected model does not exist.")
        if self.selected_framework not in self.frameworks:
            raise ValueError("selected framework does not exist.") 

        # Attempt to automatically detect some HMMs that are a good fit for the input data
        # only the first and last row of the data are used in this process
        if self.text_enable == False:         
            if (len(set(self.state_labels[0])) == 1) and (len(set(self.state_labels[-1]))):  # General Mixture Model Detection
                print("(Supervised Training Autodetect): The labels seem to remain constant, consider using the General Mixture Model.")
            elif (self.state_labels[0] == self.observations[0]) and (self.state_labels[-1] == self.observations[-1]):  # State-emission Model Detection
                print("(Supervised Training Autodetect): The states seem to emit themselves as observations, consider using the State-emission HMM.")
            else:
                print("(Supervised Training Autodetect): This appears to be a generic task, consider using Architecture B with any HMM.")

            print("Selected Architecture:", self.selected_architecture, "| Selected Model:", self.selected_model, "\n")
        else:
            print("You have opted to use additional text_data, this will require some kind of custom implementation from a scientific paper. Selected Architecture:", self.selected_architecture, "| Selected Model:", self.selected_model)

    def check_architecture_selection(self, architecture_b_algorithm):
        """
        Perform some checks regarding the selected architecture and the input data.
        """
        if self.selected_architecture == "A":
            print("Since you selected architecture 'A', you are probably not utilizing the actual truth labels of the training set and the state sequences of the test set, in this supervised task.")
            element_1 = set(self.golden_truth)
            element_2 = self.unique_states
            if element_1 != element_2 and self.selected_model == "State-emission HMM":
                raise ValueError("you have selected architecture 'A' and 'State-emission HMM' but the number of unique states is " + str(len(element_2)) + " while the number of unique truth labels is " + str(len(element_1)) + "; consider using architecture 'B'.")                  
        elif self.selected_architecture == "B":
            if architecture_b_algorithm == "forward":
                print("You have selected architecture 'B', the purely classification-based approach!. The 'forward' algorithm has some shortcomings which will be printed at the end, in this supervised task. Consider using 'formula'.")
            elif architecture_b_algorithm == "formula":
                print("You have selected architecture 'B', the purely classification-based approach!. The 'formula' algorithm is the ideal choice for supervised tasks.")

    def check_shape(self, container_1, container_2):
        """
        Given two containers, checks whether their contents are of exact same shape
        """
        if len(container_1) != len(container_2):
            return False
        for i in range(len(container_1)):
            if len(container_1[i]) != len(container_2[i]):
                return False
        return True

    def set_unique_states(self):
        """
        Find all unique state labels that occur in the entire dataset.
        """
        if (len(self.state_labels) > 0):       
            for seq in self.state_labels:
                self.unique_states.update(set(seq))
        else:
            raise ValueError("couldn't find states, the state container appears to be empty.")  

    def set_unique_states_subset(self, subset):
        """
        Find all unique state labels that occur in a specific subset.
        """
        self.unique_states_subset = set()  # Reset it
        if (len(subset) > 0):       
            for seq in subset:
                self.unique_states_subset.update(set(seq))
        else:
            raise ValueError("couldn't find states, the state container appears to be empty.") 

    def set_unique_observations_subset(self, subset):
        """
        Find all unique observations that occur in a specific subset.
        """
        self.unique_observations_subset = set()  # Reset it
        if (len(subset) > 0):       
            for seq in subset:
                self.unique_observations_subset.update(set(seq))
        else:
            raise ValueError("couldn't find observations, the state container appears to be empty.") 

    def create_state_to_label_mapping(self):
        """
        Maps the strings of states (e.g. 'pos') from the input sequence, to indices 1,2,...n that correspond to the states/matrix that the training produces.
        The mapping depends on the framework that was chosen, e.g. Pomegranate uses them in a simple sorted manner.
        """    
        if self.selected_architecture == "A":
            self.state_to_label_mapping = {}  # Simple Dict, list of Dicts is not needed
            self.state_to_label_mapping_rev = {}
            if self.selected_framework == "pome":
                for i, unique_s in enumerate(sorted(list(self.unique_states_subset))):
                    self.state_to_label_mapping[unique_s] = i 
            elif self.selected_framework == "hohmm":
                for i, unique_s in enumerate(self.trained_model.get_parameters()["all_states"]):
                    self.state_to_label_mapping[unique_s] = i        
            self.state_to_label_mapping_rev = {v: k for k, v in self.state_to_label_mapping.items()}  # Reverse the mapping so we end up with {0: "pos", 1: "neg",...}
        elif self.selected_architecture == "B":
            for current_model in self.trained_model:
                temp_mapper = {}
                temp_mapper_rev = {}
                if self.selected_framework == "pome":
                    for i, unique_s in enumerate(sorted(list(self.unique_states_subset))):
                        temp_mapper[unique_s] = i 
                elif self.selected_framework == "hohmm":
                    for i, unique_s in enumerate(current_model.get_parameters()["all_states"]):
                        temp_mapper[unique_s] = i        
                temp_mapper_rev = {v: k for k, v in temp_mapper.items()}  # Reverse the mapping so we end up with {0: "pos", 1: "neg",...}   
                self.state_to_label_mapping.append(temp_mapper)  
                self.state_to_label_mapping_rev.append(temp_mapper_rev)     

    def create_observation_to_label_mapping(self):
        """
        Maps the strings of observations (e.g. 'good') from the input sequence, to indices 1,2,...n that correspond to the matrix that the training produces.
        The mapping depends on the framework that was chosen.
        """
        if self.selected_architecture == "A":  
            self.observation_to_label_mapping = {}  # Simple Dict, list of Dicts is not needed                 
            if self.selected_framework == "pome":
                if len(self.trained_model.states) > 2:  # None-start and None-end always exist       
                    temp_dict = self.trained_model.states[0].distribution.parameters[0]  # Just the 1st no need for more
                    for i, unique_o in enumerate(temp_dict.keys()):
                        self.observation_to_label_mapping[unique_o] = i                    
                else:
                    raise ValueError("observations index to label mapping failed, the state trained object appears to be empty.")                     
            else:
                gets_obs = self.trained_model.get_parameters()["all_obs"]
                if len(gets_obs) > 0:       
                    for i, unique_o in enumerate(gets_obs):
                        self.observation_to_label_mapping[unique_o] = i 
                else:
                    raise ValueError("observations index to label mapping failed, the observation trained object appears to be empty.")
        elif self.selected_architecture == "B":
            for current_model in self.trained_model:
                temp_mapper = {}
                if self.selected_framework == "pome":
                    if len(current_model.states) > 2:  # None-start and None-end always exist       
                        temp_dict = current_model.states[0].distribution.parameters[0]  # Just the 1st no need for more
                        for i, unique_o in enumerate(temp_dict.keys()):
                            temp_mapper[unique_o] = i                    
                    else:
                        raise ValueError("observations index to label mapping failed, the state trained object appears to be empty.")                     
                else:
                    gets_obs = current_model.get_parameters()["all_obs"]
                    if len(gets_obs) > 0:       
                        for i, unique_o in enumerate(gets_obs):
                            temp_mapper[unique_o] = i 
                    else:
                        raise ValueError("observations index to label mapping failed, the observation trained object appears to be empty.")
                self.observation_to_label_mapping.append(temp_mapper)                          

    def create_hmm_to_label_mapping(self, unique_golden_truths):
        """
        Maps each model that is created, when using Architecture 'B', to the subset of data that it was trained on, e.g. data with "pos" label.
        """
        if self.selected_architecture == "B":
            for i, unique_s in enumerate(unique_golden_truths):
                self.hmm_to_label_mapping[i] = unique_s 
        else:
            raise ValueError("model to label mapping failed, please select architecture='B'.")           

    def print_probability_parameters(self):
        """
        Prints the probability matrices of the trained Hidden Markov Model.
        """
        print("# Transition probability matrix. One for each model that was trained.")
        print(self.A)
        print("# Observation probability matrix. One for each model that was trained.")
        #xxx = list(self.B[0][0, :])
        #second_smallest_value = sorted(list(set(xxx)))[1]  # Good way to find a value for formula_magic_smoothing, use with hohmm
        #print(second_smallest_value)
        print(self.B)
        print("# Initial probabilities. One for each model that was trained.")
        print(self.pi)

    def build(self, architecture, model, framework, k_fold, boosting=False,
              state_labels_pandas=[], observations_pandas=[], golden_truth_pandas=[],
              text_instead_of_sequences=[], text_enable=False, 
              n_grams=1, n_target="", n_prev_flag=False, n_dummy_flag=False, 
              pome_algorithm="baum-welch", pome_verbose=False, pome_njobs=1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,
              pome_algorithm_t="map",
              hohmm_high_order=1, hohmm_smoothing=0.0, hohmm_synthesize=False,
              architecture_b_algorithm="formula", formula_magic_smoothing=0.0
              ):
        """
        The main function of the framework. Execution starts from here.

        Parameters:
                architecture: string denoting a choice by the user.
                model: string denoting a choice by the user.
                framework: string denoting a choice by the user.
                k_fold: the number of folds to be used in the stratified cross-validation. 3 Possible mode: (1) <int> 0, disables cross-validation and enables an 80-20 split.
                                                                                                            (2) <int> 1-n, enables cross-validation with as many folds as the number's value.
                                                                                                            (3) <ndarray>, disables cross-validation and enables a split according to this array's indices.
                boosting: a boolean value that decides whether boosting (Ensemble) will be used during training.

                state_labels_pandas: pandas Series that contains the data the known sequence of state labels.
                observations_pandas: pandas Series that contains the data that will be used as observations.
                golden_truth_pandas: pandas Series that contains the actual truth labels of all instances.

                # (Placeholder for Future Implementations) #
                text_instead_of_sequences: a completely different operating mode, where the user inputs text documents;
                                           an example would be words to be used for a multivariate (multi-observation) HMM.
                                           in this scenario the first two arguments don't have to be used.
                text_enable: enables the use of the 'text_instead_of_sequences' parameter.
                #                                          #

                n_grams: n-gram order
                n_target: a string that sets the container to be used, "states", "obs" or "both".
                n_prev_flag: a boolean value that decides the behavior when a sequence is shorter than the n-gram order.
                           'True' enables the calculation of those shorter n-grams, leading to more unique states/observations.
                           'False' disables it and returns an empty list for such cases.
                n_dummy_flag: a boolean value that decides whether the length of the sequence should be maintained with the help of a dummy set.
                              e.g. on a State-emission HMM, set it to 'False' since both the states and observations get shortened.
                              However, in other scenarios where only one of the two is affected, it will end up with a shorter length per sequence.  

                pome_algorithm: refers to a setting for Pomegranate training, can be either "baum-welch", "viterbi" or "labeled". 
                pome_verbose: refers to a setting for Pomegranate training, can be either True or False.
                pome_njobs: refers to a setting for Pomegranate training, a value different than 1 enables parallelization.
                pome_smoothing_trans: refers to a setting for Pomegranate training, adds the given float to all state transitions.
                pome_smoothing_obs: refers to a setting for Pomegranate training, adds the given float to observations.

                pome_algorithm_t: refers to a setting for the prediction phase, can be either "map" or "viterbi".
        
                hohmm_high_order: refers to the order of the HMM for HOHMM training; on Pomegranate high-order can be achieved through the n-gram settings.
                hohmm_smoothing: refers to a setting for HOHMM training, adds the given float to both state transitions and observations.
                hohmm_synthesize: refers to a setting for HOHMM training, ensures to generate all permutations of states; avoids OOV and ensures model is fully ergodic.

                architecture_b_algorithm: refers to a setting for training of any framework when using architecture="B", can be either "forward", "formula"".
                formula_magic_smoothing: refers to a setting for Architecture 'B', where problematic observations (such as out-of-vocabulary values) in the prediction phase are 'treated' in an intelligent way.
        """
        self.selected_architecture = architecture
        self.selected_model = model
        self.selected_framework = framework        
        self.k_fold = k_fold

        self.check_input_type(state_labels_pandas, observations_pandas, golden_truth_pandas, text_instead_of_sequences, text_enable)
        self.verify_and_autodetect()

        if text_enable == False:
            self.length = len(self.observations)
        else:
            self.length = len(self.text_data)

        self.convert_to_ngrams_wrapper(n=n_grams, target=n_target, prev_flag=n_prev_flag, dummy_flag=n_dummy_flag, hohmm_check=(hohmm_high_order, architecture_b_algorithm))
        self.set_unique_states()

        self.check_architecture_selection(architecture_b_algorithm)

        if isinstance(k_fold, int):
            if k_fold != 0:  # Mode 2
                cross_val = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=1, random_state=random_state)
            else:  # Mode 1, Cross Validation is Disabled
                cross_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)        
        elif isinstance(k_fold, np.ndarray):  # Mode 3, Cross Validation is Disabled
            print(k_fold)

            select = np.in1d(range(self.golden_truth.shape[0]), k_fold)
            train_index = select
            test_index = ~select

            print(train_index, len(train_index))
            print(test_index, len(test_index))
        

        if True:  # Used only with IMDb dataset and with the Artificial Approach
        #for train_index, test_index in cross_val.split(self.observations, self.golden_truth):
            state_train, obs_train, y_train = self.state_labels[train_index], self.observations[train_index], self.golden_truth[train_index]  # Needs to be ndarray<list>, not list<list>
            state_test, obs_test, y_test = self.state_labels[test_index], self.observations[test_index], self.golden_truth[test_index]

            print(train_index)

            # Debug
            print("\nStates Train:", state_train[0:3])
            print("\nObs Train:", obs_train[0:3])
            print("\nGolden Truth Train:", y_train[0:3])
            print("\nLENGTH Train:", len(state_train), len(obs_train), len(y_train))
            print("\nStates Test:", state_test[0:3])
            print("\nObs Test:", obs_test[0:3])
            print("\nGolden Truth Test:", y_test[0:3])
            print("\nLENGTH Test:", len(state_test), len(obs_test), len(y_test))
            #quit()

            if boosting == True:
                self.boosting_wrapper(state_train, obs_train, y_train, state_test, obs_test, y_test, pome_algorithm, pome_verbose, pome_njobs, pome_smoothing_trans, pome_smoothing_obs, \
                                      hohmm_high_order, hohmm_smoothing, hohmm_synthesize, pome_algorithm_t, architecture_b_algorithm, formula_magic_smoothing)
            #else:
                #weights = None
            #print(weights)
            #quit()
            weights = None
        

            time_counter = time.time()
            # Training Phase
            self.set_unique_states_subset(state_train)
            self.set_unique_observations_subset(obs_train)
            self.train(state_train, obs_train, y_train, pome_algorithm, pome_verbose, pome_njobs, pome_smoothing_trans, pome_smoothing_obs, hohmm_high_order, hohmm_smoothing, hohmm_synthesize, weights)            
            # Prediction Phase
            predict = self.predict(state_test, obs_test, pome_algorithm_t, architecture_b_algorithm, formula_magic_smoothing)
            self.result_metrics(y_test, predict, time_counter)

            # Store data for future Ensembles and then reset everything
            if self.selected_architecture == "A":
                self.ensemble_stored["Mapping"].append(self.state_to_label_mapping_rev)
            elif self.selected_architecture == "B":
                self.ensemble_stored["Mapping"].append(self.hmm_to_label_mapping)
            self.ensemble_stored["Curr_Cross_Val_Golden_Truth"].append(y_test)



            self.reset()

        self.verbose_final(pome_algorithm, pome_algorithm_t, architecture_b_algorithm)
        self.clean_up()

    def train(self, state_train, obs_train, y_train, pome_algorithm, pome_verbose, pome_njobs, pome_smoothing_trans, pome_smoothing_obs, hohmm_high_order, hohmm_smoothing, hohmm_synthesize, weights):
        """
        Train a set of models using k-fold cross-validation
        """
        if self.selected_framework == 'pome':
            if pome_algorithm not in ["baum-welch", "viterbi", "labeled"]:
                raise ValueError("please set the 'pome_algorithm' parameter to one of the following: 'baum-welch', 'viterbi', 'labeled'.")
            if pome_algorithm == "labeled":
                print("--Warning: The simple counting 'labeled' algorithm is riddled with bugs and the training is going to go completely wrong, consider using 'baum-welch'.")
            if pome_njobs != 1:
                print("--Warning: the 'pome_njobs' parameter is not set to 1, which means parallelization/batch learning is enabled. Training speed will increase tremendously but accuracy might drop.")           
            if self.selected_architecture == "A":
                self._train_pome_archit_a(state_train, obs_train, None, pome_algorithm, pome_verbose, pome_njobs, pome_smoothing_trans, pome_smoothing_obs, weights)
                self.create_state_to_label_mapping()
                self.create_observation_to_label_mapping() 
                self.pome_object_to_matrices_archit_a()  # Assign to the local parameters  
            elif self.selected_architecture == "B":
                self._train_pome_archit_b(state_train, obs_train, y_train, pome_algorithm, pome_verbose, pome_njobs, pome_smoothing_trans, pome_smoothing_obs, weights)        
                self.create_state_to_label_mapping()
                self.create_observation_to_label_mapping() 
                self.pome_object_to_matrices_archit_b()  # Assign to the local parameters

        elif self.selected_framework == 'hohmm':   
            if self.selected_architecture == "A":
                self._train_hohmm_archit_a(state_train, obs_train, None, hohmm_high_order, hohmm_smoothing, hohmm_synthesize)
                self.create_state_to_label_mapping()
                self.create_observation_to_label_mapping() 
                self.hohmm_object_to_matrices_archit_a()  # Assign to the local parameters   
            elif self.selected_architecture == "B":
                self._train_hohmm_archit_b(state_train, obs_train, y_train, hohmm_high_order, hohmm_smoothing, hohmm_synthesize)  # Local parameters remain empty in this scenario  
                self.create_state_to_label_mapping()
                self.create_observation_to_label_mapping()         
                self.hohmm_object_to_matrices_archit_b()  # Assign to the local parameters  

    def _train_pome_archit_a(self, state_train, obs_train, _, pome_algorithm, pome_verbose, pome_njobs, pome_smoothing_trans, pome_smoothing_obs, weights):
        """
        Train a Hidden Markov Model using the Pomegranate framework as a baseline. 
        Architecture A is used, which is the traditional approach where a single HMM is built; even if it looks like it, it is not really suited for classification tasks.
        """
        pome_HMM = pome.HiddenMarkovModel.from_samples(pome.DiscreteDistribution, n_components=len(self.unique_states_subset), X=obs_train, labels=state_train, weights=weights,        \
                                                       algorithm=pome_algorithm, end_state=False, transition_pseudocount=pome_smoothing_trans, emission_pseudocount=pome_smoothing_obs, \
                                                       max_iterations=1, state_names=sorted(list(self.unique_states_subset)),                                                           \
                                                       verbose=pome_verbose, n_jobs=pome_njobs                                                                                          \
                                                       )
        self.trained_model = pome_HMM

    def _train_pome_archit_b(self, state_train, obs_train, y_train, pome_algorithm, pome_verbose, pome_njobs, pome_smoothing_trans, pome_smoothing_obs, weights):
        """
        Train a Hidden Markov Model using the Pomegranate framework as a baseline.
        Architecture B is used, where multiple HMMs are built, in a purely classification-based approach.
        """
        unique_golden_truths = np.unique(y_train)
        self.create_hmm_to_label_mapping(unique_golden_truths)

        index_sets = [np.where(i == y_train) for i in unique_golden_truths]
        for j in index_sets:            
            state_new, obs_new = self.validate_architecture_b_consistency(state_train[j], obs_train[j])  # Vital validation and transformation of the subsets, to avoid inconsistent number of states and observations across subsets
            pome_HMM = pome.HiddenMarkovModel.from_samples(pome.DiscreteDistribution, n_components=len(self.unique_states_subset), X=obs_new, labels=state_new, weights=weights,         \
                                                        algorithm=pome_algorithm, end_state=False, transition_pseudocount=pome_smoothing_trans, emission_pseudocount=pome_smoothing_obs, \
                                                        max_iterations=1, state_names=sorted(list(self.unique_states_subset)),                                                           \
                                                        verbose=pome_verbose, n_jobs=pome_njobs                                                                                          \
                                                        )
            self.trained_model.append(pome_HMM)  # In this scenario, we want to store a list of trained models, not just 1.

    def _train_hohmm_archit_a(self, state_train, obs_train, _, hohmm_high_order, hohmm_smoothing, hohmm_synthesize):
        """
        Train a Hidden Markov Model using the HOHMM framework as a baseline. 
        Architecture A is used, which is the traditional approach where a single HMM is built; even if it looks like it, it is not really suited for classification tasks.
        """    
        _hohmm_builder = SimpleHOHMM.HiddenMarkovModelBuilder()     
        _hohmm_builder.add_batch_training_examples(list(obs_train), list(state_train))  # The builder does not accept objects of type ndarray<list>
        _trained_hohmm = _hohmm_builder.build(highest_order=hohmm_high_order, k_smoothing=hohmm_smoothing, synthesize_states=hohmm_synthesize, include_pi=True) 
        self.trained_model = _trained_hohmm

    def _train_hohmm_archit_b(self, state_train, obs_train, y_train, hohmm_high_order, hohmm_smoothing, hohmm_synthesize):
        """
        Train a Hidden Markov Model using the HOHMM framework as a baseline.
        Architecture B is used, where multiple HMMs are built, in a purely classification-based approach.
        """
        unique_golden_truths = np.unique(y_train)
        self.create_hmm_to_label_mapping(unique_golden_truths)

        index_sets = [np.where(i == y_train) for i in unique_golden_truths]
        for j in index_sets:
            state_new, obs_new = self.validate_architecture_b_consistency(state_train[j], obs_train[j])  # Vital validation and transformation of the subsets, to avoid inconsistent number of states and observations across subsets
            _hohmm_builder = SimpleHOHMM.HiddenMarkovModelBuilder()     
            _hohmm_builder.add_batch_training_examples(list(obs_new), list(state_new))  # The builder does not accept objects of type ndarray<list>
            _trained_hohmm = _hohmm_builder.build(highest_order=hohmm_high_order, k_smoothing=hohmm_smoothing, synthesize_states=hohmm_synthesize, include_pi=True) 
            self.trained_model.append(_trained_hohmm)

    def predict(self, state_test, obs_test, pome_algorithm_t, architecture_b_algorithm, formula_magic_smoothing):
        """
        Perform predictions on new sequences.
        """
        if self.selected_framework == 'pome':
            if self.selected_architecture == "A":
                if pome_algorithm_t in ["map", "viterbi"]:
                    return(self._predict_pome_archit_a(None, obs_test, pome_algorithm_t))
                else:
                    raise ValueError("please set the 'pome_algorithm_t' parameter to one of the following: 'map', 'viterbi'.")
            elif self.selected_architecture == "B":
                if architecture_b_algorithm in ["forward", "formula"]:
                    return(self._predict_pome_archit_b(state_test, obs_test, architecture_b_algorithm, formula_magic_smoothing)) 
                else:              
                    raise ValueError("please set the 'architecture_b_algorithm' parameter to one of the following: 'forward', 'formula'.")                
        
        elif self.selected_framework == 'hohmm':
            if self.selected_architecture == "A":            
                return(self._predict_hohmm_archit_a(obs_test))
            elif self.selected_architecture == "B":
                if architecture_b_algorithm in ["forward", "formula"]:
                    return(self._predict_hohmm_archit_b(state_test, obs_test, architecture_b_algorithm, formula_magic_smoothing)) 
                else:              
                    raise ValueError("please set the 'architecture_b_algorithm' parameter to one of the following: 'forward', 'formula'.")                 

    def _predict_pome_archit_a(self, _, obs_test, pome_algorithm_t):
        """
        Performs the prediction phase when the Hidden Markov Model is based on the Pomegranate framework.
        Architecture A is used, which is the traditional approach where a single HMM is built; even if it looks like it, it is not really suited for classification tasks.
        """   
        predict_length = len(obs_test)
        total_states = len(self.unique_states_subset)
        predict = []  # The list of label predictions
        count_new_oov_local = 0 

        if pome_algorithm_t == "map":
            predict_log_proba_matrix = np.zeros((predict_length, total_states))  # The matrix of log probabilities for each label to be stored         
            for i in range(predict_length):
                if len(obs_test[i]) > 0:
                    try:      
                        temp_predict = self.trained_model.predict(obs_test[i], algorithm='map')[-1]     # We only care about the last prediction
                        temp_predict_log_proba = self.trained_model.predict_log_proba(obs_test[i])[-1]  # Using 'argmax' to not call predict twice is wrong because for random guessing all 3 probabilities are equal
                    except ValueError:  # Prediction failed probably because of out-of-vocabulary value, perform random guessing
                        count_new_oov_local += 1
                        temp_predict = random.randint(0, total_states - 1)
                        temp_predict_log_proba = [np.log(1.0 / total_states)] * total_states  # log of base e                
                else:  #  Empty sequence, perform random guessing
                    temp_predict = random.randint(0, total_states - 1)
                    temp_predict_log_proba = [np.log(1.0 / total_states)] * total_states  # log of base e

                predict.append(self.state_to_label_mapping_rev[temp_predict])
                predict_log_proba_matrix[i,:] = temp_predict_log_proba

            self.cross_val_prediction_matrix.append(predict_log_proba_matrix)
            self.count_new_oov.append(count_new_oov_local)     

        elif pome_algorithm_t == "viterbi": 
            predict_matrix = np.empty((predict_length, 1), dtype=object)  # The matrix of predictions to be stored               
            for i in range(predict_length):
                if len(obs_test[i]) > 0:
                    try:      
                        temp_predict = self.trained_model.predict(obs_test[i], algorithm='viterbi')[-1]  # We only care about the last prediction
                    except ValueError:  # Prediction failed, perform random guessing
                        count_new_oov_local += 1
                        temp_predict = random.randint(0, total_states - 1) 
                else:  #  Prediction would be pointless for an empty sequence
                    temp_predict = random.randint(0, total_states - 1) 

                predict.append(self.state_to_label_mapping_rev[temp_predict])
                predict_matrix[i] = self.state_to_label_mapping_rev[temp_predict]

            self.cross_val_prediction_matrix.append(predict_matrix)
            self.count_new_oov.append(count_new_oov_local)   

        return(predict)

    def _predict_pome_archit_b(self, state_test, obs_test, architecture_b_algorithm, formula_magic_smoothing):
        """
        Performs the prediction phase when the Hidden Markov Model is based on the Pomegranate framework.
        Architecture B is used, where multiple HMMs are built, in a purely classification-based approach. Tries to find the model that was most likely to have generated each of the instances at hand.
        """ 
        predict_length = len(obs_test)
        total_models = len(self.hmm_to_label_mapping)
        predict = []  # The list of label predictions 
        predict_log_proba_matrix = np.zeros((predict_length, total_models))  # The matrix of log probabilities for each label to be stored
        count_new_oov_local = 0 
        count_formula_problems_local = 0

        if architecture_b_algorithm == "forward":
            for i in range(predict_length): 
                # Debug
                print(i)
                temp_predict_log_proba =  []
                for current_model in self.trained_model:  # For every trained model        
                    _current_temp_log_proba = current_model.log_probability(np.array(obs_test[i]), check_input=True)  # 'check_input'=False breaks functionality completely
                                                                                                                      # Normalization already performed by Pomegranate
                    if _current_temp_log_proba == 0.0:                  # Possibly an out-of-vocabulary new observation, maybe a smarter solution would be normalized(np.log(1.0 / total_states))
                        print("--Warning: Unstable Prediction")         # It is unstable because it can be both out-of-vocabulary and perfect match, can't be sure.
                        count_new_oov_local += 1
                        _current_temp_log_proba = np.NINF                              
                    temp_predict_log_proba.append(_current_temp_log_proba)

                if len(set(temp_predict_log_proba)) == 1:  # Ensure that we don't have n equal predictions, where argmax wouldn't work
                    temp_predict = random.randint(0, total_models - 1)
                else:
                    temp_predict = np.argmax(temp_predict_log_proba)

                predict.append(self.hmm_to_label_mapping[temp_predict])
                predict_log_proba_matrix[i,:] = temp_predict_log_proba

        elif architecture_b_algorithm == "formula":
        # Formula: score = π(state1) * ObservProb(o1|state1) * P(state2|state1) * ObservProb(o2|state2) * P(state3|state2) * ...  * P(staten|staten-1) * ObservProb(on|staten) , divided by the sequence length to normalize
            for k in range(predict_length): 
                temp_predict_log_proba = []
                current_states = state_test[k]
                current_observations = obs_test[k]
                seq_length = len(current_states)

                if len(current_observations) > 0:  
                    for j in range(total_models):  # For every trained model, j is the index of current model across all containers
                        # (1.1) Transition from start to first state (pi)
                        current_state_index = self.state_to_label_mapping[j][current_states[0]]
                        current_temp_score = self.pi[j][current_state_index] 
                        # (1.2) Probability of first observation (B)
                        try:
                            current_obs_index = self.observation_to_label_mapping[j][current_observations[0]] 
                            current_temp_score *= self.B[j][current_state_index, current_obs_index]
                        except KeyError:
                            # Out-of-vocabulary value
                            count_new_oov_local += 1
                            current_temp_score *= formula_magic_smoothing                        

                        # (2) Everything that is between the first and last
                        for i in range(1, seq_length):  # This line is different between Pomegranate and HOHMM because the latter has a specific behavior for higher orders; it has multidimensional pi
                            # (2.1)
                            previous_state_index = self.state_to_label_mapping[j][current_states[i-1]] 
                            current_state_index = self.state_to_label_mapping[j][current_states[i]] 
                            try:
                                current_obs_index = self.observation_to_label_mapping[j][current_observations[i]]
                                _obsprob_temp = self.B[j][current_state_index, current_obs_index]  # Observations are on columns
                            except KeyError:
                                # Out-of-vocabulary value
                                count_new_oov_local += 1
                                _obsprob_temp = formula_magic_smoothing

                            # (2.2)
                            _trans_prob_temp = self.A[j][previous_state_index, current_state_index]
                            
                            # Score Update
                            current_temp_score = current_temp_score * _obsprob_temp * _trans_prob_temp

                        # Create a vector that contains the scores for all models
                        temp_predict_log_proba.append(np.log(current_temp_score / float(len(current_observations))))  # divided by the sequence length to normalize

                    # Comparison
                    if len(set(temp_predict_log_proba)) == 1:  # Ensure that we don't have n equal predictions, where argmax wouldn't work
                        temp_predict = random.randint(0, total_models - 1)
                        count_formula_problems_local += 1
                    else:
                        temp_predict = np.argmax(temp_predict_log_proba)

                else:  #  Prediction would be pointless for an empty sequence
                    temp_predict = random.randint(0, total_models - 1)            
                    temp_predict_log_proba = [np.NINF] * total_models  # or could do what I do on 'map' algorithm with log
        
                predict.append(self.hmm_to_label_mapping[temp_predict])
                predict_log_proba_matrix[k,:] = temp_predict_log_proba                  

                # Debug
                # print(temp_predict_log_proba, self.hmm_to_label_mapping)
                # print(predict)
                # print(predict_log_proba_matrix)   
                # quit() 
            
            self.count_formula_problems.append(count_formula_problems_local)    
        
        self.cross_val_prediction_matrix.append(predict_log_proba_matrix)
        self.count_new_oov.append(count_new_oov_local)

        return(predict)

    def _predict_hohmm_archit_a(self, obs_test):
        """
        Performs the prediction phase when the Hidden Markov Model is based on the HOHMM framework.
        Architecture A is used, which is the traditional approach where a single HMM is built; even if it looks like it, it is not really suited for classification tasks.
        """    
        predict_length = len(obs_test) 
        total_states = len(self.unique_states_subset)
        predict = []  # The list of label predictions 
        count_new_oov_local = 0         
        predict_matrix = np.empty((predict_length, 1), dtype=object)  # The matrix of predictions to be stored             
        for i in range(predict_length):
            if len(obs_test[i]) > 0:  
                try:              
                    temp_predict = self.trained_model.decode(obs_test[i])[-1]  # We only care about the last prediction; should be of <list> type for HOHMM to work
                except ValueError:  # Prediction failed probably because of out-of-vocabulary value, perform random guessing
                    count_new_oov_local += 1
                    temp_predict = self.state_to_label_mapping_rev[random.randint(0, total_states - 1)]
            else:  #  Empty sequence, perform random guessing
                temp_predict = self.state_to_label_mapping_rev[random.randint(0, total_states - 1)]

            predict.append(temp_predict)  # This framework outputs the label name not an index
            predict_matrix[i] = temp_predict

        self.cross_val_prediction_matrix.append(predict_matrix)
        self.count_new_oov.append(count_new_oov_local)

        return(predict)   

    def _predict_hohmm_archit_b(self, state_test, obs_test, architecture_b_algorithm, formula_magic_smoothing):
        """
        Performs the prediction phase when the Hidden Markov Model is based on the HOHMM framework.
        Architecture B is used, where multiple HMMs are built, in a purely classification-based approach. Tries to find the model that was most likely to have generated each of the instances at hand.
        """ 
        predict_length = len(obs_test)
        total_models = len(self.hmm_to_label_mapping)
        predict = []  # The list of label predictions 
        predict_log_proba_matrix = np.zeros((predict_length, total_models))  # The matrix of log probabilities for each label to be stored         
        count_new_oov_local = 0 
        count_formula_problems_local = 0

        if architecture_b_algorithm == "forward":
            for i in range(predict_length):   
                # Debug
                print(i)                                  
                temp_predict_log_proba =  []
                for j in range(total_models):  # For every trained model         
                    try:
                        _current_temp_log_proba = np.log(self.trained_model[j].evaluate(obs_test[i]))  # Unsure if normalization is performed
                    except ValueError:  # Catches both empty sequences and out-of-vocabulary scenarios
                        if len(obs_test[i]) > 0:
                            count_new_oov_local += 1
                        _current_temp_log_proba = np.NINF                                                       
                    temp_predict_log_proba.append(_current_temp_log_proba)

                if len(set(temp_predict_log_proba)) == 1:  # Ensure that we don't have n equal predictions, where argmax wouldn't work
                    temp_predict = random.randint(0, total_models - 1)
                else:
                    temp_predict = np.argmax(temp_predict_log_proba)

                predict.append(self.hmm_to_label_mapping[temp_predict])
                predict_log_proba_matrix[i,:] = temp_predict_log_proba

        elif architecture_b_algorithm == "formula":
        # IMPOSSIBLE FOR HIGH-ORDERS BECAUSE (self.B[j]) for HOHMM remains the same for all orders   
            # detect_high_order = len(self.pi)
            # object_hohmm = SimpleHOHMM.HiddenMarkovModelBuilder()
            # new_state_test = object_hohmm._make_higher_order_states(state_sequences=state_test, order=detect_high_order) 
                # if len(current_observations) > 0:  
                #     for j in range(total_models):  # For every trained model, j is the index of current model across all containers
                #         # ! HOHMM takes a different approach than Pomegranate when it comes to pi and higher-order
                #         current_pi = current_states[0].split('-')
                #         current_temp_score = 1.0
                #         for pi_index in range(detect_high_order):
                #             # (1.1) Transitions from start to one or more states at the start of the sequence (pi), e.g. seq=["pos-neg", "pos-pos", ...], means we should do start->"pos" and start->"pos-neg"
                #             temp = '-'.join(current_pi[0:pi_index+1])
                #             current_temp_score *= self.pi[pi_index][temp]

                #         current_state_index = self.state_to_label_mapping[j][current_states[0]]
                #         current_obs_index = self.observation_to_label_mapping[j][current_observations[0]] 

                #         # (1.2) Probability of first observation (B)
                #         current_temp_score *= self.B[j][current_state_index, current_obs_index]

        # Formula: score = π(state1) * ObservProb(o1|state1) * P(state2|state1) * ObservProb(o2|state2) * P(state3|state2) * ...  * P(staten|staten-1) * ObservProb(on|staten) , divided by the sequence length to normalize
            for k in range(predict_length): 
                # Debug
                #print(k)                                 
                temp_predict_log_proba = []
                current_states = state_test[k]
                current_observations = obs_test[k]
                seq_length = len(current_states)

                if len(current_observations) > 0:  
                    for j in range(total_models):  # For every trained model, j is the index of current model across all containers
                        # (1.1) Transition from start to first state (pi)
                        current_state_index = self.state_to_label_mapping[j][current_states[0]]
                        current_temp_score = self.pi[j][0][current_states[0]] 
                        # (1.2) Probability of first observation (B)
                        try:
                            current_obs_index = self.observation_to_label_mapping[j][current_observations[0]] 
                            current_temp_score *= self.B[j][current_state_index, current_obs_index]
                        except KeyError:
                            # Out-of-vocabulary value
                            count_new_oov_local += 1
                            current_temp_score *= formula_magic_smoothing                       

                        # (2) Everything that is between the first and last
                        for i in range(1, seq_length):  # This line is different between Pomegranate and HOHMM because the latter has a specific behavior for higher orders; it has multidimensional pi
                            # (2.1)
                            previous_state_index = self.state_to_label_mapping[j][current_states[i-1]] 
                            current_state_index = self.state_to_label_mapping[j][current_states[i]] 
                            try:
                                current_obs_index = self.observation_to_label_mapping[j][current_observations[i]]
                                _obsprob_temp = self.B[j][current_state_index, current_obs_index]  # Observations are on columns
                            except KeyError:
                                # Out-of-vocabulary value
                                count_new_oov_local += 1
                                _obsprob_temp = formula_magic_smoothing

                            # (2.2)
                            _trans_prob_temp = self.A[j][previous_state_index, current_state_index]
                            
                            # Score Update
                            current_temp_score = current_temp_score * _obsprob_temp * _trans_prob_temp

                        # Create a vector that contains the scores for all models
                        temp_predict_log_proba.append(np.log(current_temp_score / float(len(current_observations))))  # divided by the sequence length to normalize

                    # Comparison
                    if len(set(temp_predict_log_proba)) == 1:  # Ensure that we don't have n equal predictions, where argmax wouldn't work
                        temp_predict = random.randint(0, total_models - 1)
                        count_formula_problems_local += 1
                    else:
                        temp_predict = np.argmax(temp_predict_log_proba)

                else:  #  Prediction would be pointless for an empty sequence
                    temp_predict = random.randint(0, total_models - 1)            
                    temp_predict_log_proba = [np.NINF] * total_models  # or could do what I do on 'map' algorithm with log
        
                predict.append(self.hmm_to_label_mapping[temp_predict])
                predict_log_proba_matrix[k,:] = temp_predict_log_proba                  

                # Debug
                # print(temp_predict_log_proba, self.hmm_to_label_mapping)
                # print(predict)
                # print(predict_log_proba_matrix)   
                # quit() 
            
            self.count_formula_problems.append(count_formula_problems_local)    
        
        self.cross_val_prediction_matrix.append(predict_log_proba_matrix)
        self.count_new_oov.append(count_new_oov_local)


        return(predict)

    def convert_to_ngrams_wrapper(self, n, target, prev_flag, dummy_flag, hohmm_check):
        """
        (1) Execute the n-gram conversion on the correct container, as defined by 'target'.
        (2) Perform some input data validation checks.
        """
        if self.selected_framework == "hohmm":
            if hohmm_check[0] > 1 and hohmm_check[1] == "formula":
                raise ValueError("the selected 'hohmm' framework combined with 'formula' algorithm has not been implemented for high-order HMMs.")

        if n < 2:
            print("N-gram conversion is disabled.")
            return None
        if target not in ["states", "obs", "both"]:
            raise ValueError("invalid selection of target for the n-gram process.")
        if self.selected_framework != "pome":
            if target != "obs":
                raise ValueError("you should be attempting to perform n-grams on the states only when using the 'pome' framework.")

        if target == "states":
            self._convert_to_ngrams(n, "states", prev_flag, dummy_flag)
        elif target == "obs":
            self._convert_to_ngrams(n, "obs", prev_flag, dummy_flag)
        elif target == "both":
            self._convert_to_ngrams(n, "states", prev_flag, dummy_flag)
            self._convert_to_ngrams(n, "obs", prev_flag, dummy_flag)
        print("N-gram conversion to", n, "\b-gram was sucessful. Container type remained as ndarray<list>.")

        if self.check_shape(self.state_labels, self.observations) == False:
            print("--Warning: one of the containers is now shorter than the other on the y axis, consider using the flags or using 'both' as target.")
            self._ngrams_balance_shape()

    def _convert_to_ngrams(self, n, target, prev_flag, dummy_flag):
        """
        Convert the contents of a single container to an n-gram representation.

        Parameters:
                n: n-gram order.
                container: the container of data, on which to perform the conversion.
                target: a string setting that can take the following values, "states", "obs" or "both".
                prev_flag: a boolean value that decides the behavior when a sequence is shorter than the n-gram order.
                           'True' enables the calculation of those shorter n-grams, leading to more unique states/observations.
                           'False' disables it and returns an empty list for such cases.
                dummy_flag: a boolean value that decides whether the length of the sequence should be maintained with the help of a dummy set.
                            e.g. on a State-emission HMM, set it to 'False' since both the states and observations get shortened.
                                 However, in other scenarios where only one of the two is affected, it will end up with a shorter length per sequence.
        """
        if target == "states":        
            container = self.state_labels  # Shallow copy not Deep, be careful
        elif target == "obs":
            container = self.observations
        
        if (len(container) > 0):
            ngrams_temp = []
            for seq in container:
                seq_deep = copy.deepcopy(seq)   # Insert is going to be used, better make a deep copy
                current_seq = list()
                if len(seq_deep) >= n:
                    if dummy_flag == True:
                        for i in range(n-1):  # Append one or more dummies at the start of the sequence
                            seq_deep.insert(i, "dummy"+str(i))
                    for grams in ngramsgenerator(seq_deep, n):
                        current_seq.append("".join(grams))  

                elif prev_flag == True:
                    if dummy_flag == True:
                        for i in range(len(seq_deep)-1):
                            seq_deep.insert(i, "dummy"+str(i))
                    for grams in ngramsgenerator(seq_deep, len(seq_deep)):
                        current_seq.append("".join(grams))                    

                ngrams_temp.append(current_seq)  

            if target == "states":
                self.state_labels = np.array(ngrams_temp)
            elif target == "obs":
                self.observations = np.array(ngrams_temp)
        else:
            raise ValueError("n-gram conversion failed, the input container appears to be empty.")          

    def _ngrams_balance_shape(self):
        """
        Executed after the n-gram process is completed, if the state and observation containers of the class are not of the same exact shape.
        (1) If a row is empty on one container, wipes it on the other. Caused by prev_flag=False.
        (2) If a row is shorter on one container, removes the first elements on the other. Caused by dummy_flag=False. This is probably not the best idea, but we are all consenting adults here.
        """
        count_wipe = 0
        count_shorten = 0
        for i in range(self.length):
            length_1 = len(self.state_labels[i])
            length_2 = len(self.observations[i])
            if length_1 == 0:
                self.observations[i] = []
                count_wipe += 1
            elif length_2 == 0:
                self.state_labels[i] = []
                count_wipe += 1

            length_1 = len(self.state_labels[i])
            length_2 = len(self.observations[i])
            if length_1 > length_2:
                self.state_labels[i] = self.state_labels[i][-length_2:]  # Remove the first elements of the list
                count_shorten += 1
            elif length_2 > length_1:
                self.observations[i] = self.observations[i][-length_1:]  # Remove the first elements of the list
                count_shorten += 1    

        if count_wipe > 0:
            print("---Wiped", count_wipe, "rows/instances. Caused by prev_flag=False.")
        if count_shorten > 0:
            print("---Shortened", count_shorten, "rows/instances. Caused by dummy_flag=False. This is probably not the best idea, but we are all consenting adults here.")
        if self.check_shape(self.state_labels, self.observations) == False:
            raise ValueError("one of the containers is still shorter than the other on the y axis.")               
        else:
            print("--Containers are now of exact same shape")

    def boosting_wrapper(self, state_train, obs_train, y_train, state_test, obs_test, y_test, pome_algorithm, pome_verbose, pome_njobs, pome_smoothing_trans, pome_smoothing_obs, hohmm_high_order, hohmm_smoothing, hohmm_synthesize, pome_algorithm_t, architecture_b_algorithm, formula_magic_smoothing):
        """
        (1) Executes the Kang et al.'s Boosting implementation from the Ensemble_Framework.py.
        (2) The approach is AdaBoost preceded by Bootstrap selection. The Bootstrap process is required to artifically drop the performance because we don't have a Decision Tree here, the performance would be 96% from the start.
        (3) Manages the training and evaluation phase since they are different compared to the traditional scenario.
        """

        # THE ENTIRE ALGORITHM FROM THE PAPER IS POINTLESS, IT DOESN'T MINIMIZE AN ERROR FUNCTION, INSTEAD IT KEEP SELECTING RANDOMLY AND MISCLASIFYING THE SAME INSTANCES. STEP 3-e. doesn't even make sense

        if self.selected_framework == "pome" and self.selected_architecture == "A" and pome_algorithm_t == "viterbi":
            raise ValueError("for 'viterbi' on Architecture A we can't perform this type of boosting since we have no log probabilities available.")
        
        precomputed_seeds = [22, 55, 62, 11]
        iterations = 40
        train_length = len(y_train)
        final_predict_on_train_or_test = "train"        

        cross_val_prediction_matrix_final = []
        mapping = []
        golden_truth = []          

        print("\nInitiating AdaBoost with Bootstraping. Iterations:", iterations, "\n")

        # 0.
        weights_A = np.empty(iterations)  # refers to the weight of each classifier on the Final Ensemble

        # 1.
        t = 0
        # 2. Initialize weights
        weights_D = np.ones(train_length) / train_length  # refers to the weight of each sample on AdaBoost

        # 3.
        while t < iterations:
            time_counter = time.time()        
            # 3-a. Bootstrap/Bagging with replacement
            #np.random.seed(seed=precomputed_seeds[t])    
            np.random.seed()     
            random_sample_ind = np.random.choice(np.arange(train_length), size=int(train_length*0.28), p=weights_D, replace=False)  # replace refers to something else here
                                                                                                                       # this matrix can also operate as a mapping
            random_sample_state = state_train[random_sample_ind]
            random_sample_obs = obs_train[random_sample_ind]
            random_y = y_train[random_sample_ind]

            #train_cross_val_matrix_boost =  
            #test_cross_val_matrix =
            #print(np.sort(random_sample_ind))
            #print()
            #quit()
            #print(state_train[0])
            #print()
            #print(state_train[random_sample_ind].shape)
            #print(random_sample_state[np.where(random_sample_ind == 0)[0][0]])

            # 3-b. Training Phase
            self.set_unique_states_subset(state_train)
            self.set_unique_observations_subset(obs_train)
            self.train(random_sample_state, random_sample_obs, random_y, pome_algorithm, pome_verbose, pome_njobs, pome_smoothing_trans, pome_smoothing_obs, hohmm_high_order, hohmm_smoothing, hohmm_synthesize, None)
            
            # 3-c. Prediction Phase on the TEST DATA 
            predict = self.predict(state_test, obs_test, pome_algorithm_t, architecture_b_algorithm, formula_magic_smoothing)
            self.result_metrics(y_test, predict, time_counter)
            # Not an Ensemble, just storing the log probability matrix #
            if self.selected_architecture == "A":
                self.ensemble_stored["Mapping"].append(self.state_to_label_mapping_rev)
            elif self.selected_architecture == "B":
                self.ensemble_stored["Mapping"].append(self.hmm_to_label_mapping)
            if final_predict_on_train_or_test == "train":
                self.ensemble_stored["Curr_Cross_Val_Golden_Truth"].append(y_train)  
            elif final_predict_on_train_or_test == "test":
                self.ensemble_stored["Curr_Cross_Val_Golden_Truth"].append(y_test)                      
            cross_val_prediction_matrix_final.append(self.cross_val_prediction_matrix)
            mapping.append(self.ensemble_stored["Mapping"]) 
            golden_truth.append(self.ensemble_stored["Curr_Cross_Val_Golden_Truth"])     
            #                                         #

            # 3-c. Prediction Phase on the TRAINING DATA
            predict = self.predict(state_train, obs_train, pome_algorithm_t, architecture_b_algorithm, formula_magic_smoothing)
            self.result_metrics(y_train, predict, time_counter)   

            # 3-d.
            y_product = np.empty(train_length)
            incorrect_pred = []
            for i, comparison in enumerate(zip(y_train, predict)):
                if comparison[0] == comparison [1]:                
                    y_product[i] = 1.0
                else:
                    y_product[i] = -1.0 
                    incorrect_pred.append(weights_D[i])

            # Normally would be
            #err = 1 - self.cross_val_metrics['Accuracy'][1]
            err = np.sum(incorrect_pred)
            print("err:", err, "outo-of-vocabulary:", self.count_new_oov, "formula problems:", self.count_formula_problems)

            # 3-d. Verbose
            print("\n[" + str(t+1) + "] Current Accuracy on the entire Train set: " + str(self.cross_val_metrics['Accuracy'][1] * 100) + " and on the Test set: " + str(self.cross_val_metrics['Accuracy'][0] * 100))
 
            # Debug
            #print(len(cross_val_prediction_matrix_final))
            #print(self.cross_val_prediction_matrix[0][0:20])
            #print(weights_A)
            #quit()
            # 3-e.
            if err > 0.5:
                continue

            # 3-f.
            weights_A[t] = 0.5 * np.log((1 - err) / err)

            # 3-g.              
            weights_D *= np.exp(-1.0 * weights_A[t] * y_product)
            #print(weights_D)
            #print(np.exp(-1.0 * weights_A[t] * y_product))
            #print(y_product)
            # weights_D *= np.exp(1 * y_product)
            # print(weights_D)

            # 3-h.
            weights_D = weights_D / np.sum(weights_D)

            # Reset literally everything
            self.reset()
            self.cross_val_metrics = defaultdict(list)
            self.cross_val_prediction_matrix = []
            self.count_new_oov = []
            self.count_formula_problems = []  

            # 3-i.
            t = t + 1
        
        print(weights_A)
        quit()

        for i in range(len(cross_val_prediction_matrix_final)):
            if final_predict_on_train_or_test == "train":  # This axis is originally meant for cross validation folds but here it is used to store train_matrix on [0] and test_matrix on [1]
                cross_val_prediction_matrix_final[i].pop(0)
            elif final_predict_on_train_or_test == "test":
                cross_val_prediction_matrix_final[i].pop(1)

        # print(len(cross_val_prediction_matrix_final[0]))
        # quit()     
        # print(len(cross_val_prediction_matrix_final[0]))
        # print(cross_val_prediction_matrix_final[0][0].shape)
            
        # AT THE END
        Ensemble_Framework.ensemble_run(cross_val_prediction_matrix_final, mapping, golden_truth, mode="sum", weights=list(weights_A), use_log_prob=False, detailed=True)
        quit()


    def adaboost_multiclass_classic(self, state_train, obs_train, y_train, state_test, obs_test, y_test, pome_algorithm, pome_verbose, pome_njobs, pome_smoothing_trans, pome_smoothing_obs, hohmm_high_order, hohmm_smoothing, hohmm_synthesize, pome_algorithm_t, architecture_b_algorithm, formula_magic_smoothing):
        discrete_or_log = "log"

        """Implement a single boost using the SAMME.R which is based on probabilities and can work on multi-class.
        Inspired by https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/weight_boosting.py, Multi-class AdaBoost (2006)"""

        iterations = 50
        learning_rate = 1.0
        from scipy.special import xlogy

        if self.selected_framework == "pome" and self.selected_architecture == "A" and pome_algorithm_t == "viterbi":
            print("--Warning: For 'viterbi' on Architecture A you are limited to discrete boosting since we have no log probabilities available.")
            discrete_or_log = "discrete"
        elif self.selected_framework == "hohmm":
            raise ValueError("boosting is not supported for 'hohmm' yet, select 'pome'.") 

        train_length = len(y_train)
        classes = list(set(y_train))
        n_classes = len(classes)

        # Initialize weights
        sample_weight = np.ones(train_length) / train_length

        for i in range(iterations):
            time_counter = time.time()
            # Training Phase
            self.set_unique_states_subset(state_train)
            self.set_unique_observations_subset(obs_train)
            self.train(state_train, obs_train, y_train, pome_algorithm, pome_verbose, pome_njobs, pome_smoothing_trans, pome_smoothing_obs, hohmm_high_order, hohmm_smoothing, hohmm_synthesize, sample_weight)
            # Prediction Phase on the TRAINING DATA specifically
            predict = self.predict(state_train, obs_train, pome_algorithm_t, architecture_b_algorithm, formula_magic_smoothing)
            self.result_metrics(y_train, predict, time_counter)

            predict_log_proba = self.cross_val_prediction_matrix[0]

            print(predict_log_proba)

            # Instances incorrectly classified
            incorrect = y_train != predict

            # Error fraction
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

            # Stop if classification is perfect
            if estimator_error <= 0:
                break

            print(estimator_error, self.cross_val_metrics["Accuracy"][0])
            print(np.where(sample_weight > 0.00025013))

            # Construct y coding as described in Zhu et al [2]:
            #
            #    y_k = 1 if c == k else -1 / (K - 1)
            #
            # where K == n_classes_ and c, k in [0, K) are indices along the second axis of the y coding with c being the index corresponding to the true class label.
            y_codes = np.array([-1. / (n_classes - 1), 1.])
            y_coding = y_codes.take(classes == y_train[:, np.newaxis])

            # Displace zero probabilities so the log is defined.
            # Also fix negative elements which may occur with negative sample weights.
            proba = predict_log_proba  # alias for readability
            np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

            # Boost weight using multi-class AdaBoost SAMME.R alg
            estimator_weight = (-1. * learning_rate * ((n_classes - 1.) / n_classes) * xlogy(y_coding, proba).sum(axis=1))

            # Only boost the weights if it will fit again
            if not i == iterations - 1:
                # Only boost positive weights
                sample_weight *= np.exp(estimator_weight * ((sample_weight > 0) | (estimator_weight < 0)))

            # Reset
            self.cross_val_metrics = defaultdict(list)
            self.reset()

        return sample_weight
        #Ensemble_Framework.adaboost()


    def pome_object_to_matrices_archit_a(self):
        """
        Assigns the A, B and pi matrices with the values from a Pomegranate trained object/model (Architecture A). They should be ndarrays.
        """
        # Note that last state is 'None-End' and previous to last is 'None-Start'
        # A
        self.A = self.trained_model.dense_transition_matrix()[:-2,:-2]  # All except last 2
        # B
        remaining_states = self.trained_model.states[:-2]               # All except last 2
        temp_obs_matrix = np.zeros((len(self.state_to_label_mapping), len(self.observation_to_label_mapping)))
        for i, row in enumerate(remaining_states):         
            temp_obs_matrix[i, :] = list(row.distribution.parameters[0].values())     
        self.B = temp_obs_matrix
        # pi
        self.pi = self.trained_model.dense_transition_matrix()[-2,:-2]  # Previous to last

    def pome_object_to_matrices_archit_b(self):
        """
        Assigns the A, B and pi matrices with the values from multiple Pomegranate trained objects/models (Architecture B). They should be ndarrays.
        """
        self.A = []  # Reset it for each fold of the cross validation, since we are using append constantly
        self.B = []
        self.pi = []
        for i, current_model in enumerate(self.trained_model):
            # A
            self.A.append(current_model.dense_transition_matrix()[:-2,:-2])  # All except last 2
            # B
            remaining_states = current_model.states[:-2]               # All except last 2
            temp_obs_matrix = np.zeros((len(self.state_to_label_mapping[i]), len(self.observation_to_label_mapping[i])))
            for i, row in enumerate(remaining_states):         
                temp_obs_matrix[i, :] = list(row.distribution.parameters[0].values())     
            self.B.append(temp_obs_matrix)
            # pi
            self.pi.append(current_model.dense_transition_matrix()[-2,:-2])  # Previous to last

    def hohmm_object_to_matrices_archit_a(self):
        """
        Assigns the A, B and pi matrices with the values from a HOHMM trained object/model (Architecture A). They should be ndarrays.
        """
        get_params = self.trained_model.get_parameters()
        # A
        self.A = np.array(get_params["A"])
        # B
        self.B = np.array(get_params["B"])
        # pi - stored in a specific Dict format, used for higher order HOHMM-approaches, needed for 'formula' algorithm
        self.pi = get_params["pi"]
        # self.pi = np.array(list(get_params["pi"][0].values()))

    def hohmm_object_to_matrices_archit_b(self):
        """
        Assigns the A, B and pi matrices with the values from multiple HOHMM trained object/model (Architecture B). They should be ndarrays.
        """
        self.A = []  # Reset it for each fold of the cross validation, since we are using append constantly
        self.B = []
        self.pi = []
        for current_model in self.trained_model:
            get_params = current_model.get_parameters()
            # A
            self.A.append(np.array(get_params["A"]))
            # B
            self.B.append(np.array(get_params["B"]))
            # pi - stored in a specific Dict format, used for higher order HOHMM-approaches, needed for 'formula' algorithm
            self.pi.append(get_params["pi"])
            # self.pi.append(np.array(list(get_params["pi"][0].values())))

    def validate_architecture_b_consistency(self, state_subset, obs_subset):
        """
        If we have a very imbalanced or small dataset, it is obvious that trying one HMM on each subset of data will lead to inconsistent number
        of states and observations across them. I suggest to add a new instance with the states/observations that are missing.
        """
        # States
        validate_x = set()     
        for seq in state_subset:
            validate_x.update(set(seq))

        difference_s = self.unique_states_subset.difference(validate_x)
        if len(difference_s) > 0:
            print("--Warning: The input dataset is way too small or imbalanced to facilitate splitting it with Architecture B. Adding the states that are missing using new instances...")
            state_subset = list(state_subset)  # Inefficient but I want to maintain type of ndarray<list>
            obs_subset = list(obs_subset)            
            for missing in difference_s:
                state_subset.append([missing])
                obs_subset.append(random.sample(self.unique_observations_subset, 1))  # Add a random instance to the other container
            state_subset = np.array(state_subset) 
            obs_subset = np.array(obs_subset)           

        # Observations
        validate_x = set()     
        for seq in obs_subset:
            validate_x.update(set(seq))

        difference_o = self.unique_observations_subset.difference(validate_x)
        if len(difference_s) > 0:
            print("--Warning: The input dataset is way too small or imbalanced to facilitate splitting it with Architecture B. Adding the observations that are missing using new instances...")
            state_subset = list(state_subset)  # Inefficient but I want to maintain type of ndarray<list>
            obs_subset = list(obs_subset)            
            for missing in difference_o:
                obs_subset.append([missing])
                state_subset.append(random.sample(self.unique_states_subset, 1))  # Add a random instance to the other container
            state_subset = np.array(state_subset) 
            obs_subset = np.array(obs_subset) 

        return(state_subset, obs_subset)

    def result_metrics(self, golden_truth, prediction, time_counter):
        """
        Assigns a local variable (dict) with the resulting metrics of the current fold of the cross validation.
        """
        # Metrics
        accuracy = metrics.accuracy_score(golden_truth, prediction)
        rest_as_string = metrics.classification_report(golden_truth, prediction, output_dict=False)  # Used as a string
        rest_as_dict = metrics.classification_report(golden_truth, prediction, output_dict=True)  # Used as an information source
        confusion_matrix = metrics.confusion_matrix(golden_truth, prediction)     

        self.cross_val_metrics["Name"].append(self.selected_model)
        self.cross_val_metrics["F1-score"].append(rest_as_dict['weighted avg']['f1-score'])
        self.cross_val_metrics["Accuracy"].append(accuracy)
        self.cross_val_metrics["Metrics_String"].append("- - - - - RESULT METRICS - " + self.selected_model + " - - - -\nExact Accuracy: " + str(accuracy) + "\n" + rest_as_string)
        self.cross_val_metrics["Confusion_Matrix"].append(confusion_matrix)
        self.cross_val_metrics["Time_Complexity"].append(time.time() - time_counter)                         

    def print_best_results(self, detailed=True, decimals=5):
        """   
        Prints the best results across all folds of the cross validation    
        """
        if len(self.cross_val_metrics["Name"]) > 0:
            print("\n", "- " * 18, "Conclusion across", str(self.k_fold), "\b-fold Cross Validation", "- " * 18,)
            index_1, index_2 = np.argmax(self.cross_val_metrics["F1-score"]), np.argmax(self.cross_val_metrics["Accuracy"])
            if detailed == True:            
                print("-Best F1-score:\n", self.cross_val_metrics["Metrics_String"][index_1], "\n", self.cross_val_metrics["Confusion_Matrix"][index_1], sep='')
                print("\n-Best Accuracy score:\n", self.cross_val_metrics["Metrics_String"][index_2], "\n", self.cross_val_metrics["Confusion_Matrix"][index_2], "\n", sep='')
            print("-Best F1-score:", np.around(self.cross_val_metrics["F1-score"][index_1]*100.0, decimals=decimals))
            print("-Best Accuracy:", np.around(self.cross_val_metrics["Accuracy"][index_2]*100.0, decimals=decimals))
        else:
            raise ValueError("cannot print best results; it seems that no predictions have been performed.")               

    def print_average_results(self, decimals=5):
        """   
        Prints the average results across all folds of the cross validation    
        """
        if len(self.cross_val_metrics["Name"]) > 0:
            print("\n", "- " * 18, "Conclusion across", str(self.k_fold), "\b-fold Cross Validation", "- " * 18,)
            print("-Average F1-score:", np.around(np.mean(self.cross_val_metrics["F1-score"])*100.0, decimals=decimals))
            print("-Average Accuracy:", np.around(np.mean(self.cross_val_metrics["Accuracy"])*100.0, decimals=decimals))
        else:
            raise ValueError("cannot print average results; it seems that no predictions have been performed.")               
 
    def verbose_final(self, pome_algorithm, pome_algorithm_t, architecture_b_algorithm):
        """
        Verbose after the entire training and prediction is completed in order to inform the user.
        """
        if self.selected_framework == "pome":
            if self.selected_architecture == "A":
                print("State to label mapping completed using sorting ('pome'-based). x", self.k_fold, "times")  
                print("Observation to label mapping completed ('pome'-based). x", self.k_fold, "times")
            elif self.selected_architecture == "B":        
                print("HMM model to label mapping completed. x", self.k_fold, "times")                    
            if pome_algorithm == "baum-welch":
                print(self.k_fold, "\b-fold cross validation completed using the Baum-Welch algorithm for training. Since this algorithm is originally meant for un/semi-supervised scenarios 'max_iterations' was set to 1.") 
            elif pome_algorithm == "viterbi":
                print(self.k_fold, "\b-fold cross validation completed using the Viterbi algorithm for training. Consider using 'baum-welch' since it is better. Additionaly, it is worth noting that both these algorithms are originally meant for un/semi-supervised scenarios.")   
            elif pome_algorithm == "labeled":
                print(self.k_fold, "\b-fold cross validation completed using the Labeled algorithm for training.") 
            if self.selected_architecture == "A":                        
                if pome_algorithm_t == "map":
                    print("Prediction was performed using the Maximum a Posteriori algorithm. Returns a set of log probabilities, stored in 'cross_val_prediction_matrix'.")
                elif pome_algorithm_t == "viterbi":
                    print("Prediction was performed using the Viterbi algorithm. Returns a set of exact predictions, not probabilities, stored in 'cross_val_prediction_matrix'.") 
            elif self.selected_architecture == "B": 
                if architecture_b_algorithm == "forward":                
                    print("Prediction was performed using the Forward algorithm on multiple models. It does not utilize the state sequences of the test set and has no special function for out-of-vocabulary values, consider using 'formula'. Returns a set of log probabilities, stored in 'cross_val_prediction_matrix'.")
                elif architecture_b_algorithm == "formula": 
                    print("Prediction was performed using the Formula algorithm on multiple models. Returns a set of log probabilities, stored in 'cross_val_prediction_matrix'.")
        if self.selected_framework == "hohmm": 
            if self.selected_architecture == "A":
                print("State to label mapping completed ('hohmm'-based). x", self.k_fold, "times")  
                print("Observation to label mapping completed ('hohmm'-based). x", self.k_fold, "times") 
            elif self.selected_architecture == "B":        
                print("HMM model to label mapping completed. x", self.k_fold, "times")  
            print(self.k_fold, "\b-fold cross validation completed using the Labeled algorithm for training.") 
            if self.selected_architecture == "A":                        
                print("Prediction was performed using the Viterbi algorithm. Returns a set of exact predictions, not probabilities, stored in 'cross_val_prediction_matrix'.")                
            elif self.selected_architecture == "B": 
                if architecture_b_algorithm == "forward":                
                    print("Prediction was performed using the Forward algorithm on multiple models. It does not utilize the state sequences of the test set and has no special function for out-of-vocabulary values, consider using 'formula'. Returns a set of log probabilities, stored in 'cross_val_prediction_matrix'.")
                elif architecture_b_algorithm == "formula": 
                    print("Prediction was performed using the Formula algorithm on multiple models. Returns a set of log probabilities, stored in 'cross_val_prediction_matrix'.")             
        print("Predictions failed and random guessing was performed because of out-of-vocabulary new observations, on an average of:", np.mean(np.array(self.count_new_oov)), "observations.")
        if self.selected_architecture == "B" and architecture_b_algorithm == "formula":
            print("Predictions failed and random guessing was performed because the 'formula' encountered a problem, on an average of:", np.mean(np.array(self.count_formula_problems)), "instances.")
             
def plot_vertical(x_f1, x_acc, y, dataset_name, k_fold):
    """
    Given two lists of performance metrics and one list of strings, constructs a high quality vertical comparison chart.
    """
    length = len(y)
    if any(len(lst) != length for lst in [x_f1, x_acc, y]):
        raise ValueError("the two lists of values and one list of strings must all be of exact same length")

    indices = np.arange(length)
            
    fig, ax1 = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(left=0.18, top=0.92, bottom=0.08)
    fig.canvas.set_window_title(dataset_name + " - Averages across " + str(k_fold) + "-fold Cross Validation")

    p1 = ax1.bar(indices, x_f1, align="center", width=0.35, color="cornflowerblue")
    p2 = ax1.bar(indices + 0.35, x_acc, align="center", width=0.35, color="navy")  

    ax1.set_title(dataset_name + " - Averages across " + str(k_fold) + "-fold Cross Validation")
    ax1.set_ylim([0, 1])
    ax1.yaxis.set_major_locator(MaxNLocator(11))
    ax1.yaxis.grid(True, linestyle='--', which="major", color="grey", alpha=.25)
    ax1.legend((p1[0], p2[0]), ("F1-score", "Accuracy"))

    ax1.set_ylabel("Performance")   
    
    ax1.set_xticks(indices + 0.35 / 2)
    ax1.set_xticklabels(y)

    # Rotate labels and align them horizontally to the left 
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")

    # Automatically adjust subplot parameters so that the the subplot fits in to the figure area
    fig.tight_layout()

    plt.show()

def plot_horizontal(x_f1, x_acc, y, dataset_name, k_fold):
    """
    Given two lists of performance metrics and one list of strings, constructs a high quality horizontal comparison chart.
    """
    length = len(y)
    if any(len(lst) != length for lst in [x_f1, x_acc, y]):
        raise ValueError("the two lists of values and one list of strings must all be of exact same length")

    indices = np.arange(length)
             
    # Reverse the items to appear in correct order
    x_f1.reverse()
    x_acc.reverse()
    y.reverse()

    fig, ax1 = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(left=0.18, top=0.92, bottom=0.08)
    fig.canvas.set_window_title(dataset_name + " - Averages across " + str(k_fold) + "-fold Cross Validation")

    p1 = ax1.barh(indices, x_f1, align="center", height=0.35, color="cornflowerblue")
    p2 = ax1.barh(indices - 0.35, x_acc, align="center", height=0.35, color="navy")    
 
    ax1.set_title(dataset_name + " - Averages across " + str(k_fold) + "-fold Cross Validation")
    ax1.set_xlim([0, 1])
    ax1.xaxis.set_major_locator(MaxNLocator(11))
    ax1.xaxis.grid(True, linestyle='--', which="major", color="grey", alpha=.25)
    ax1.legend((p1[0], p2[0]), ("F1-score", "Accuracy"))

    ax1.set_yticks(indices - 0.35 / 2)
    ax1.set_yticklabels(y)

    # Right-hand Y-axis
    indices_new = list(itertools.chain.from_iterable((indices[m], indices[m] - 0.35) for m in range(length)))  # Trick to print text on the y axis for both bars
    set_decimals = 4
    rounded_f1 = list(np.around(np.array(x_f1), set_decimals))
    rounded_acc = list(np.around(np.array(x_acc), set_decimals))

    ax2 = ax1.twinx()
    ax2.set_yticks(indices_new)
    ax2.set_ylim(ax1.get_ylim())  # Make sure that the limits are set equally on both yaxis so the ticks line up
    ax2.set_yticklabels(n for n in itertools.chain.from_iterable(itertools.zip_longest(rounded_f1, rounded_acc)) if n)  # Combine two lists in an alternating fashion
    ax2.set_ylabel("Performance rounded to " + str(set_decimals) + " decimals")

    # Automatically adjust subplot parameters so that the the subplot fits in to the figure area
    fig.tight_layout()

    plt.show()

def plot_basic(x, y, dataset_name, k_fold, y_text):
    """
    Given one list and a list of strings, constructs a high quality vertical comparison chart.
    """
    indices = np.arange(len(y))

    fig, ax1 = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(left=0.18, top=0.92, bottom=0.08)
    fig.canvas.set_window_title(dataset_name + " - Averages across " + str(k_fold) + "-fold Cross Validation")

    ax1.bar(indices, x, align="center", width=0.35, color="navy") 

    ax1.set_title(dataset_name + " - Averages across " + str(k_fold) + "-fold Cross Validation")
    ax1.yaxis.set_major_locator(MaxNLocator(11))
    ax1.yaxis.grid(True, linestyle='--', which="major", color="grey", alpha=.25)

    ax1.set_ylabel(y_text)  # e.g. Time (sec)  
    
    ax1.set_xticks(indices)
    ax1.set_xticklabels(y)

    # Rotate labels and align them horizontally to the left 
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")

    # Automatically adjust subplot parameters so that the the subplot fits in to the figure area
    fig.tight_layout()

    plt.show()

def pome_graph_plot(pomegranate_model):
    """
    Given a trained Pomegranate model, plots the Hidden Markov Model Graph.
    """
    fig, ax1 = plt.subplots()
    fig.canvas.set_window_title("Hidden Markov Model Graph")
    ax1.set_title("Hidden Markov Model")
    pomegranate_model.plot()
    plt.show()

def general_mixture_model_label_generator(sequences, individual_labels):
    """
    Given a pandas Series of sequences and one label per sequence, 'multiplies' the label in order to have
    a single constant label per sequence and outputs it as a pandas Series.
    """
    sequences = sequences.values
    individual_labels = individual_labels.values

    transformed_labels = list()
    for i, seq in enumerate(sequences):
        transformed_labels.append([individual_labels[i]] * len(seq))

    return(pd.Series(transformed_labels))
    