# Takes into consideration the previous observation for each time step instead of the current one and doesn't even spot "whateveromegalul"
# algorithm='baum-welch' fixes the bug, 
# removing state_names on the 1st example also fixes the bug but introduces errors
# e.g. hmm_supervised_pos = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=3, X=pos_data_corresponding_to_labels, algorithm='labeled', verbose=True, labels=pos_clustered_labeled_data, n_jobs=1)

from pomegranate import *
import numpy as np

# Happens in this example
pos_clustered_labeled_data = [["dummy1", "pos", "neg", "neg", "dummy1"], ["dummy1", "pos"]]
pos_data_corresponding_to_labels = [["dummy1", "good", "bad", "bad", "whateveromegalul"], ["dummy1", "good"]]

# New Idea
#print(hmm_supervised_pos.states.index(hmm_supervised_pos.states[2]))
#print(hmm_supervised_pos.dense_transition_matrix()[:-2,:-2])
# from_matrix function

distribution = DiscreteDistribution

d1 = DiscreteDistribution.from_samples(pos_data_corresponding_to_labels[0])
d2 = DiscreteDistribution.from_samples(pos_data_corresponding_to_labels[0])
d3 = DiscreteDistribution.from_samples(pos_data_corresponding_to_labels[0])

print(type(d1), d1)

s1 = State(d1, name="dummy1")
s2 = State(d2, name="pos")
s3 = State(d3, name="neg")

model = HiddenMarkovModel('example')
model.add_states([s1, s2, s3])
model.add_transition(model.start, s1, 0.33)
model.add_transition(model.start, s2, 0.33)
model.add_transition(model.start, s3, 0.33)
model.add_transition(s1, s1, 0.33)
model.add_transition(s1, s2, 0.33)
model.add_transition(s1, s3, 0.33)
model.add_transition(s2, s1, 0.33)
model.add_transition(s2, s2, 0.33)
model.add_transition(s2, s3, 0.33)
model.add_transition(s3, s1, 0.33)
model.add_transition(s3, s2, 0.33)
model.add_transition(s3, s3, 0.33)
model.bake(merge = "All")


state_names = [state.name for state in model.states]
state_indices = [[state_names.index(s) for s in seq] for seq in pos_clustered_labeled_data]
state_values = [[model.states[i] for i in seq] for seq in state_indices]

model.fit(sequences=pos_data_corresponding_to_labels, algorithm='labeled', verbose=True, labels=state_values, n_jobs=1)
#quit()
print("NEXT HMM")
for i in list(model.states):
    print(i.name)
print(model)

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
fig.canvas.set_window_title("Hidden Markov Model Graph")
model.plot()
plt.show()
#quit()

pos_clustered_labeled_data = [["dummy1", "pos", "neg", "neg", "dummy1"], ["dummy1", "pos"]]
pos_data_corresponding_to_labels = [["dummy1", "good", "bad", "bad", "whateveromegalul"], ["dummy1", "good"]]

# Training
# Build Pos Class HMM - !!! state_names should be in alphabetical order
hmm_supervised_pos = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=3, X=pos_data_corresponding_to_labels, algorithm='labeled', verbose=True, labels=pos_clustered_labeled_data, n_jobs=1, state_names=sorted(["pos", "neg", "dummy1"]))

print("NEXT HMM")
for i in list(hmm_supervised_pos.states):
    print(i.name)
print(hmm_supervised_pos)
print("edw")
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