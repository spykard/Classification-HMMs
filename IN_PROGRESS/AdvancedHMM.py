""" 
AdvancedHMM: A framework that implements (mostly supervised) state-of-the-art Hidden Markov Models for classfication tasks.
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class HiddenMarkovModel:
    """
        observations: Observation sequence
        state_labels: Hidden State sequence
        A: Transition probability matrix
        B: Observation probability matrix
        pi: Initial probabilities
    """

def plot_vertical(x_f1, x_acc, y, dataset_name, k_fold):
    """
    Given two lists of performance metrics and one list of strings, constructs a high quality vertical comparison chart.
    """
    length = len(y)
    if any(len(lst) != length for lst in [x_f1, x_acc, y]):
        raise ValueError("the two lists of values and one list of strings must all be of same length")

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
        raise ValueError("the two lists of values and one list of strings must all be of same length")

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