""" 
AdvancedHMM: A framework that implements state-of-the-art Hidden Markov Models, mostly for supervised/classification tasks.
"""

import numpy as np
import itertools
from collections import defaultdict
import pandas as pd
import pomegranate as pome
import time  # Pomegranate has it's own 'time' and can cause conflicts
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from nltk import ngrams as ngramsgenerator

class AdvancedHMM:
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
        self.A = None
        self.B = None
        self.pi = None

        self.text_data = []
        self.text_enable = False        
        self.length = 0
        self.architectures = ["A", "B"]
        self.selected_architecture = ""
        self.models = ["General Mixture Model", "State-emission HMM", "Classic HMM"]
        self.selected_model = ""
        self.frameworks = ["pome", "hohmm"]
        self.selected_framework = ""

        self.trained_model = None  # The object that is outputed after training on the current cross validation fold; depends on the framework that was used
        self.multivariate = False
        self.state_to_label_mapping = {}

        # Evaluation
        self.k_fold = 0
        self.cross_val_metrics = defaultdict(list)  # {Name: [], 
                                                    # F1-score: [], 
                                                    # Accuracy: [], 
                                                    # Metrics_String: []], 
                                                    # Time_Complexity: []}

    def clean_up(self):
        """
        Resets the values which are not needed to their original values, after the entire process is finished.
        """
        self.state_labels = []
        self.observations = []
        self.text_data = []        
        self.trained_model = None

    def check_input_type(self, state_labels_pandas, observations_pandas, text_instead_of_sequences, text_enable):
        """
        Make sure that the input data are of the correct type: pandas Series. Also perform the assignment to the local variables.
        """
        self.text_enable = text_enable
        if self.text_enable == False:
            if (isinstance(state_labels_pandas, pd.Series) != True) or (isinstance(observations_pandas, pd.Series) != True):
                raise ValueError("please make sure that you are inputting the parameters 'state_labels_pandas' and 'observations_pandas' in the form of a pandas Series, i.e. select a column of a DataFrame.")
            else:
                self.state_labels = state_labels_pandas.values  # or tolist() to get list instead of ndarray
                self.observations = observations_pandas.values 
        else:
            if (isinstance(text_instead_of_sequences, pd.Series) != True):  # just text parameter
                raise ValueError("please make sure that you are inputting the parameter 'text_instead_of_sequences' in the form of a pandas Series, i.e. select a column of a DataFrame.")
            else:
                self.text_data = text_instead_of_sequences.values           
            if len(state_labels_pandas) > 0:  # text + state_labels parameters
                if (isinstance(state_labels_pandas, pd.Series) != True):
                    raise ValueError("text parameter was given correctly but please make sure that you are inputting the parameters 'state_labels_pandas' in the form of a pandas Series, i.e. select a column of a DataFrame.")
                else:
                    self.state_labels = state_labels_pandas.values
            if len(observations_pandas) > 0:  # text + observations parameters
                if (isinstance(observations_pandas, pd.Series) != True):
                    raise ValueError("text parameter was given correctly but please make sure that you are inputting the parameter 'observations_pandas' in the form of a pandas Series, i.e. select a column of a DataFrame.")               
                else:
                    self.observations = observations_pandas.values 

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

    def set_mapping(self):
        """
        Depends on the framework that is being used
        """
        print("inprogress")
        # self.state_to_label_mapping
        # observation_mapping = set()
        # for obs_lst in pos_data_corresponding_to_labels:
        #     observation_mapping.update(set(obs_lst))


    def build(self, architecture, model, framework, k_fold, state_labels_pandas=[], observations_pandas=[], text_instead_of_sequences=[], text_enable=False, n_grams=1, n_target="", n_prev_flag=False, n_dummy_flag=False):
        """
        The main function of the framework. Execution starts from here.

        Parameters:
                architecture: string denoting a choice by the user.
                model: string denoting a choice by the user.
                framework: string denoting a choice by the user.
                k_fold: the number of folds to be used in the cross-validation.

                state_labels_pandas: pandas Series that contains the data that will be used as labels for the states.
                observations_pandas: pandas Series that contains the data that will be used as observations.

                text_instead_of_sequences: a completely different operating mode, where the user inputs text documents; 
                                           in this scenario the first two arguments don't have to be used.
                text_enable: enables the use of the 'text_instead_of_sequences' parameter.

                n_grams: n-gram order
                n_target: a string that sets the container to be used, "states", "obs" or "both".
                n_prev_flag: a boolean value that decides the behavior when a sequence is shorter than the n-gram order.
                           'True' enables the calculation of those shorter n-grams, leading to more unique states/observations.
                           'False' disables it and returns an empty list for such cases.
                n_dummy_flag: a boolean value that decides whether the length of the sequence should be maintained with the help of a dummy set.
                            e.g. on a State-emission HMM, set it to 'False' since both the states and observations get shortened.
                                 However, in other scenarios where only one of the two is affected, it will end up with a shorter length per sequence.                
        """
        self.selected_architecture = architecture
        self.selected_model = model
        self.selected_framework = framework        
        self.k_fold = k_fold

        self.check_input_type(state_labels_pandas, observations_pandas, text_instead_of_sequences, text_enable)
        self.verify_and_autodetect()

        if text_enable == False:
            self.length = len(self.observations)
        else:
            self.length = len(self.text_data)

        self.convert_to_ngrams(n=n_grams, target=n_target, prev_flag=n_prev_flag, dummy_flag=n_dummy_flag)

        # Train


    def train(self):
        time_counter = time.time()

        pome_HMM = pome.HiddenMarkovModel.from_samples(pome.DiscreteDistribution, len(documentSentiments), X=data_train_transformed, labels=labels_supervised, state_names=state_names, n_jobs=n_jobs, verbose=False)
        if verbose == True:
            for x in range(0, len(documentSentiments)):
                print("State", hmm_leanfrominput_supervised.states[x].name, hmm_leanfrominput_supervised.states[x].distribution.parameters)
        print("Indexes:", tuple(zip(documentSentiments, state_names)))
        ### Plot the Hidden Markov Model Graph
        if graph_print_enable == 1:
            fig, ax1 = plt.subplots()
            fig.canvas.set_window_title("Hidden Markov Model Graph")
            ax1.set_title(str(n_order) + "-th Order")
            hmm_leanfrominput_supervised.plot()
            plt.show()
        ###

    def convert_to_ngrams(self, n, target, prev_flag, dummy_flag):
        """
        Convert the contents of containers to an n-gram representation.

        Parameters:
                n: n-gram order
                target: a string that sets the container to be used, "states", "obs" or "both".
                prev_flag: a boolean value that decides the behavior when a sequence is shorter than the n-gram order.
                           'True' enables the calculation of those shorter n-grams, leading to more unique states/observations.
                           'False' disables it and returns an empty list for such cases.
                dummy_flag: a boolean value that decides whether the length of the sequence should be maintained with the help of a dummy set.
                            e.g. on a State-emission HMM, set it to 'False' since both the states and observations get shortened.
                                 However, in other scenarios where only one of the two is affected, it will end up with a shorter length per sequence.
        """
        if n < 2:
            print("N-gram conversion is disabled.")
            return None
        if target not in ["states", "obs", "both"]:
            raise ValueError("invalid selection of target for the n-gram process.")
        if self.selected_framework != "pome":
            if target != "obs":
                raise ValueError("you should be attempting to perform n-grams on the states only when using the 'pome' framework.")   

        if (len(self.state_labels) > 0) and (len(self.observations) > 0):
            ngrams_temp = []
            for seq in self.observations:
                current_seq = list()
                if len(seq) >= n:
                    if dummy_flag == True:
                        for i in range(n-1):  # Append one or more dummies at the start of the sequence
                            seq.insert(i, "dummy"+str(i+1))
                    for grams in ngramsgenerator(seq, n):
                        current_seq.append("".join(grams))  

                elif prev_flag == True:
                    if dummy_flag == True:
                        for i in range(len(seq)-1):
                            seq.insert(i, "dummy"+str(i+1))
                    for grams in ngramsgenerator(seq, len(seq)):
                        current_seq.append("".join(grams))                    

                ngrams_temp.append(current_seq)  

            self.observations = ngrams_temp
            print("N-gram conversion to", n, "\b-gram was sucessful. Container type was also changed from ndarray<list> to list<list>.")
        else:
            raise ValueError("n-gram conversion failed, one or both of the state/observations containers appear to be empty.")          

        if self.check_shape(self.state_labels, self.observations) == False:
            print("--Warning: one of the containers is now shorter than the other on the y axis, consider using the flags or using 'both' as target.")
            self._ngrams_balance_shape()

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
            if length_1 > length_2:
                self.state_labels[i] = self.state_labels[i][-length_2:]  # Remove the first elements of the list
                count_shorten += 1
            elif length_2 > length_1:
                self.observations[i] = self.observations[i][-length_1:]  # Remove the first elements of the list
                count_shorten += 1    

        if count_wipe > 0:
            print("---Wiped", count_wipe, "rows/instances. Caused by prev_flag=False.")
        if count_shorten > 0:
            print("---Shorterned", count_shorten, "rows/instances. Caused by dummy_flag=False. This is probably not the best idea, but we are all consenting adults here.")
        if self.check_shape(self.state_labels, self.observations) == False:
            raise ValueError("one of the containers is still shorter than the other on the y axis.")               
        else:
            print("--Containers are now of exact same shape")

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

def general_mixture_model_label_generator(sequences, individual_labels):
    """
    Given a pandas Series of sequences and one label per sequence, 'multiplies' the label in order to have
    a single constant label per sequence and outputs it as a pandas Series
    """
    sequences = sequences.values
    individual_labels = individual_labels.values

    transformed_labels = list()
    for i, seq in enumerate(sequences):
        #getlength = len(data_train_transformed[i])
        #state_name = "s" + str(documentSentiments.index(x))
        transformed_labels.append([individual_labels[i]] * len(seq))

    return(pd.Series(transformed_labels))


# labels = [["dummy1", "pos", "neg", "neg", "dummy1"], ["dummy1", "pos"]]
# observations = [["dummy1", "good", "bad", "bad", "whateveromegalul"], ["dummy1", "good"]]
# labels_series = pd.Series(labels)
# observations_series = pd.Series(observations)
# hmm = AdvancedHMM()
# hmm.build(architecture="A", model="State-emission HMM", k_fold=1, state_labels_pandas=labels_series, observations_pandas=observations_series, )


labels = [["dummy1", "pos", "pos", "pos"], ["dummy1", "pos"]]
observations = [["1", "2", "3", "4"], ["1", "1"]]
labels_series = pd.Series(labels)
observations_series = pd.Series(observations)
hmm = AdvancedHMM()
# text = pd.Series(["omegalol", "omeg"])
hmm.build(architecture="A", model="State-emission HMM", framework="pome", k_fold=1,   \
          state_labels_pandas=labels_series, observations_pandas=observations_series, \
          text_instead_of_sequences=[], text_enable=False,                            \
          n_grams=1, n_target="obs", n_prev_flag=False, n_dummy_flag=False)
