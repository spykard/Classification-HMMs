import numpy as np
from pomegranate import *
from scipy.misc import logsumexp
from SimpleHOHMM import HiddenMarkovModelBuilder as Builder

x1 = np.array([['normal', 'cold', 'dizzy', 'dizzy','normal','normal'], ['normal', 'cold', 'dizzy', 'dizzy','normal','normal']])
x2 = np.array([['cold', 'cold', 'dizzy', 'normal','normal','normal'], ['cold', 'cold', 'dizzy', 'normal','normal','normal']])
#x3 = np.array([['dizzy', 'dizzy', 'cold', 'normal', 'dizzy', 'normal'], ['dizzy', 'dizzy', 'cold', 'normal', 'dizzy', 'normal']])
#x4 = np.array([['normal', 'normal', 'cold', 'dizzy', 'dizzy', 'dizzy'], ['normal', 'normal', 'cold', 'dizzy', 'dizzy', 'dizzy']])

s1 = ['healthy', 'healthy', 'fever', 'fever', 'healthy', 'healthy']
s2 = ['healthy', 'fever', 'fever', 'healthy', 'healthy', 'fever']
#s3 = np.array([['normal', 'cold', 'dizzy', 'dizzy','normal','normal'], ['normal', 'cold', 'dizzy', 'dizzy','normal','normal']])
#s4 =np.array([['normal', 'cold', 'dizzy', 'dizzy','normal','normal'], ['normal', 'cold', 'dizzy', 'dizzy','normal','normal']])


# states = [
#         ['healthy', 'healthy', 'fever', 'fever', 'healthy', 'healthy'],
#         ['healthy', 'fever', 'fever', 'healthy', 'healthy', 'fever'],
#         ['fever', 'fever', 'fever', 'healthy', 'healthy', 'healthy'],
#         ['healthy', 'healthy', 'healthy', 'fever', 'fever', 'fever'],
#         ['fever', 'fever', 'fever', 'fever', 'fever', 'healthy'],
#         ['healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy'],
#         ['healthy', 'healthy', 'healthy', 'fever', 'fever', 'healthy'],
# ]

# text = [w.split() for w in ["this DT",
#                             "is V",
#                             "a DT",
#                             "test N",
#                             "for IN",
#                             "a DT",
#                             "hidden Adj",
#                             "Markov N",
#                             "model N"]]
# words, y = zip(*text)
# vocab, identities = np.unique(words, return_inverse=True)
# X = (identities.reshape(-1, 1) == np.arange(len(vocab))).astype(int)

# print(X,vocab)
# quit()

# print(X)
# classes, y = np.unique(y, return_inverse=True)
# Y = y.reshape(-1, 1) == np.arange(len(classes))
# feature_prob = np.log(np.dot(Y.T, X))
# feature_prob -= logsumexp(feature_prob, axis=0)

# quit()

# xx = [x1,x2]
# lab = [s1,s2]

#print(X)
# X_concat = np.concatenate(X)

# if X_concat.ndim == 1:
#         X_concat = X_concat.reshape(X_concat.shape[0], 1)

# n, d = X_concat.shape

#print(X_concat)
#print(n, d)
#print(sequence[0][1])

# xx = IndependentComponentsDistribution([NormalDistribution(0,1), NormalDistribution(0,1), NormalDistribution(0,1), NormalDistribution(0,1)])
# print(xx)



# Multivariate Continuous, in a TFIDF scenario -> sequence [[object1_tfidf_of_word1, object1_tfidf_of_word2...], [object2_tfidf_of_word1, object2_tfidf_of_word2...]]
emis = [  # Let's make it so that the 3rd feature means healthy
        [[0.20, 0.10, 0.95, 0.50], [0.25, 0.10, 0.00, 0.50], [0.25, 0.10, 0.00, 0.50], [0.25, 0.10, 0.00, 0.50]],  # Seq 1
        [[0.25, 0.10, 0.10, 0.50], [0.00, 0.10, 0.10, 0.50], [0.25, 0.10, 0.00, 0.50], [0.25, 0.10, 0.00, 0.50]],  # Seq 2
        [[0.30, 0.55, 0.80, 0.50], [0.50, 0.55, 0.10, 0.50], [0.25, 0.10, 0.00, 0.50], [0.25, 0.10, 0.00, 0.50]],  # Seq 3
        [[0.60, 0.55, 0.10, 0.50], [0.20, 0.15, 0.10, 0.50], [0.25, 0.10, 0.00, 0.50], [0.25, 0.10, 0.00, 0.50]]   # Seq 4
        ]
trans = [["healthy","fever","fever","fever"], ["fever","fever","fever","fever"], ["healthy","healthy","fever","fever"], ["fever","healthy","fever","fever"]]  # ADD 1 MORE LABEL THAN SEQUENCES AND IT GIVES ERROR, ADD 1 LESS AND IT ENABLES SEMI-SUPERVISED TRAINING

# emis = [  # Let's make it so that the 3rd feature means healthy
#         [0.20, 0.10, 0.95],  # Seq 1
#         [0.25, 0.10, 0.10],  # Seq 2
#         [0.30, 0.55, 0.80],  # Seq 3
#         [0.60, 0.55, 0.10]   # Seq 4
#         ]
# trans = [["healthy","fever","fever"], ["fever","fever","fever"], ["healthy","healthy","fever"], ["fever","healthy","fever"]]  # ADD 1 MORE LABEL THAN SEQUENCES AND IT GIVES ERROR, ADD 1 LESS AND IT ENABLES SEMI-SUPERVISED TRAINING

# https://github.com/jmschrei/pomegranate/issues/120
distrs = [NormalDistribution(0.5,1), NormalDistribution(0.5,1), NormalDistribution(0.5,1), NormalDistribution(0.5,1)]
distrs = [PoissonDistribution(0,1), PoissonDistribution(0,1), PoissonDistribution(0,1), PoissonDistribution(0,1)]
#create_ICD = IndependentComponentsDistribution(distrs)
# IT WORKSSSSS!
#x = IndependentComponentsDistribution.from_samples([[11, 22, 33, 44], [11, 22, 33, 44]], distributions=distrs)  # Can only use One-Hot encoding as numbers
# X = emis
# labels = trans
# X_ = [x for x, label in zip(X, labels) if label != None]
# X_ = np.concatenate(X_) 
# labels_ = np.concatenate([l for l in labels if l is not None])
# x = [IndependentComponentsDistribution.from_samples(X_[labels_ == "healthy"], distributions=distrs)]
# print(x)
# quit()

hmm_leanfrominput = HiddenMarkovModel.from_samples(IndependentComponentsDistribution, 2, X=emis, labels=trans, independent_dists=distrs)
print(hmm_leanfrominput)
quit()

#print(hmm_leanfrominput.states)
#for x in range(0, 2):
#        print("State", hmm_leanfrominput.states[x].name, hmm_leanfrominput.states[x].distribution.parameters)

# Multivariate DISCRETE
emis = [  # Let's make it so that the 3rd feature means healthy
        [["c1", "c2", "c2"], ["c1", "c2", "c2"]],  # Seq 1
        [["c1", "c2", "c2"], ["c1", "c2", "c2"]],  # Seq 2
        [["c1", "c2", "c2"], ["c1", "c2", "c2"]],  # Seq 3
        [["c1", "c2", "c2"], ["c1", "c2", "c2"]],   # Seq 4
        ]
     
trans = [["healthy","fever"], ["fever","fever"], ["healthy","healthy"], ["fever","healthy"]]  # ADD 1 MORE LABEL THAN SEQUENCES AND IT GIVES ERROR, ADD 1 LESS AND IT ENABLES SEMI-SUPERVISED TRAINING

d1 = DiscreteDistribution({"just_initializing": 1.00})
d2 = DiscreteDistribution({"just_initializing": 1.00})
d3 = DiscreteDistribution({"just_initializing": 1.00})

# DOESN'T WORK WITH DISCREETE ANYWAY https://github.com/jmschrei/pomegranate/issues/402
distri_sum = [d1,d2,d3]
x = IndependentComponentsDistribution.from_samples([[11, 22, 33]], distributions=distri_sum)  # Can only use One-Hot encoding as numbers
x = IndependentComponentsDistribution.from_samples([["lol", "lol", "lol"]], distributions=distri_sum)  # Can only use One-Hot encoding as numbers
print(x)

# IndependentComponentsDistribution from_samples, Can only use numbers in One-Hot encoding for strings
# SEE GITHUB ISSUE POST FOR FIX
hmm_leanfrominput = HiddenMarkovModel.from_samples(IndependentComponentsDistribution, 2, X=emis, labels=trans, independent_dists=[DiscreteDistribution, DiscreteDistribution, DiscreteDistribution])
#hmm_leanfrominput = HiddenMarkovModel.from_samples(IndependentComponentsDistribution([DiscreteDistribution({"c1": 0.33, "c2": 0.33, "c3": 0.33}), DiscreteDistribution({"c1": 0.33, "c2": 0.33, "c3": 0.33})]), 2, X=emis, labels=trans)


print(hmm_leanfrominput.states)
#for x in range(0, 2):
#        print("State", hmm_leanfrominput.states[x].name, hmm_leanfrominput.states[x].distribution.parameters)




quit()
# modeld = 2
# n = len(sequence)
# sequence_ndarray = numpy.empty((n, modeld), dtype=numpy.float64)

# for i in range(n):
#         for j in range(modeld):
#                 symbol = sequence[i][j]
#                 keymap = model.keymap[j]

#                 if isinstance(symbol, str) and symbol == 'nan':
#                         sequence_ndarray[i, j] = numpy.nan
#                 elif isinstance(symbol, (int, float)) and numpy.isnan(symbol):
#                         sequence_ndarray[i, j] = numpy.nan
#                 elif symbol in keymap:
#                         sequence_ndarray[i, j] = keymap[symbol]
#                 else:
#                         raise ValueError("Symbol '{}' is not defined in a distribution"
#                         .format(symbol))


# print(sequence_ndarray)
#quit()




observations = [
        ['normal', 'cold', 'dizzy', 'dizzy','normal','normal'],
        ['cold', 'cold', 'dizzy', 'normal','normal','normal'],
        ['dizzy', 'dizzy', 'cold', 'normal', 'dizzy', 'normal'],
        ['normal', 'normal', 'cold', 'dizzy', 'dizzy', 'dizzy'],
        ['cold', 'cold', 'cold', 'cold', 'normal', 'normal'],
        ['dizzy', 'dizzy', 'normal', 'dizzy', 'normal', 'dizzy'],
        ['normal', 'normal', 'dizzy', 'cold', 'dizzy', 'normal']
]

states = [
        ['healthy', 'healthy', 'fever', 'fever', 'healthy', 'healthy'],
        ['healthy', 'fever', 'fever', 'healthy', 'healthy', 'fever'],
        ['fever', 'fever', 'fever', 'healthy', 'healthy', 'healthy'],
        ['healthy', 'healthy', 'healthy', 'fever', 'fever', 'fever'],
        ['fever', 'fever', 'fever', 'fever', 'fever', 'healthy'],
        ['healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy'],
        ['healthy', 'healthy', 'healthy', 'fever', 'fever', 'healthy'],
]


builder = Builder()
#builder.add_batch_training_examples(observations, states)
#hmm = builder.build(highest_order = 5)
hmm = builder.build_unsupervised(single_states=['healthy', 'fever'], all_obs=['normal', 'cold', 'dizzy'], distribution="random", highest_order=4)
hmm.display_parameters()
hmm.learn(observations, k_smoothing=0.001)
hmm.display_parameters()

# obs =  ['normal', 'cold', 'dizzy']
# states = hmm.decode(obs)
# print(states) # prints: ['healthy', 'healthy', 'fever']

# obs = ['normal', 'cold', 'dizzy']
# likelihood = hmm.evaluate(obs)
# print(likelihood) # prints: 0.0433770021525