# Takes into consideration the previous observation for each time step instead of the current one and doesn't even spot "whateveromegalul"
# algorithm='baum-welch' fixes the bug, 
# removing state_names on the 1st example also fixes the bug but introduces errors
# e.g. hmm_supervised_pos = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=3, X=pos_data_corresponding_to_labels, algorithm='labeled', verbose=True, labels=pos_clustered_labeled_data, n_jobs=1)

from pomegranate import *

# Happens in this example
pos_clustered_labeled_data = [["dummy1", "pos", "neg", "neg", "dummy1"], ["dummy1", "pos"]]
pos_data_corresponding_to_labels = [["dummy1", "good", "bad", "bad", "whateveromegalul"], ["dummy1", "good"]]

# builder = Builder()
# builder.add_batch_training_examples(pos_data_corresponding_to_labels, pos_clustered_labeled_data)
# hmm = builder.build()
# hmm.display_parameters()
# quit()

# Training
# Build Pos Class HMM - !!! state_names should be in alphabetical order
hmm_supervised_pos = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=3, X=pos_data_corresponding_to_labels, algorithm='labeled', verbose=True, labels=pos_clustered_labeled_data, n_jobs=1, state_names=sorted(["pos", "neg", "dummy1"]))

print("NEXT HMM")
for i in list(hmm_supervised_pos.states):
    print(i.name)
print(hmm_supervised_pos)
quit()

# Also happens in this example without using state_names
pos_clustered_labeled_data = [["s0", "s1", "s2", "s2", "s0"], ["s0", "s1"]]
pos_data_corresponding_to_labels = [["dummy1", "good", "bad", "bad", "whateveromegalul"], ["dummy1", "good"]]

# builder = Builder()
# builder.add_batch_training_examples(pos_data_corresponding_to_labels, pos_clustered_labeled_data)
# hmm = builder.build()
# hmm.display_parameters()
# quit()

# Training
# Build Pos Class HMM - !!! state_names should be in alphabetical order
hmm_supervised_pos = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=3, X=pos_data_corresponding_to_labels, algorithm='labeled', verbose=True, labels=pos_clustered_labeled_data, n_jobs=1)
print("NEXT HMM")
for i in list(hmm_supervised_pos.states):
    print(i.name)
print(hmm_supervised_pos)
quit()