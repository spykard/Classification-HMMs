# Python 3

from pomegranate import *
import matplotlib.pyplot as plt
import math

# Doctor Example
seq = list('NCDDDDDDDDDNNNN')

d1 = DiscreteDistribution({'D': 0.1, 'C': 0.4, 'N': 0.5})
d2 = DiscreteDistribution({'D': 0.6, 'C': 0.3, 'N': 0.1})

s1 = State( d1, name='Healthy' )
s2 = State( d2, name='Fever' )

hmm = HiddenMarkovModel()
hmm.add_states(s1, s2)
hmm.add_transition( hmm.start, s1, 0.6 )
hmm.add_transition( hmm.start, s2, 0.4 )
hmm.add_transition( s1, s1, 0.7 )
hmm.add_transition( s1, s2, 0.3 )
hmm.add_transition( s2, s1, 0.4 )
hmm.add_transition( s2, s2, 0.6 )
hmm.bake()

hmm_predictions = hmm.predict( seq )

print ("sequence: {}".format( ''.join( seq ) ))
print ("hmm pred: {}".format( ''.join( map( str, hmm_predictions ) ) ))

hmm_predictions = hmm.predict( seq, algorithm='viterbi' )[1:-1]

print ("sequence: {}".format( ''.join( seq ) ))
print ("hmm pred: {}".format( ''.join( map( str, hmm_predictions ) ) ))

# END #

# Doctor Example - Machine Learning the parameters

print ("1 for fit() / 2 for from_samples ()")
z = input()

        # Initialize
if (z=='1'):
    X = list()
    for i in range(0, 100):
        reroll = hmm.sample(10000, False)  
        X.append(reroll)

    d11 = DiscreteDistribution({'D': 0.15, 'C': 0.15, 'N': 0.7})
    d22 = DiscreteDistribution({'D': 0.35, 'C': 0.35, 'N': 0.15})

    s11 = State( d11, name='Healthy' )
    s22 = State( d22, name='Fever' )

    hmm_leanfrominput = HiddenMarkovModel()
    hmm_leanfrominput.add_states(s11, s22)
    hmm_leanfrominput.add_transition( hmm_leanfrominput.start, s11, 0.5 )
    hmm_leanfrominput.add_transition( hmm_leanfrominput.start, s22, 0.5 )
    hmm_leanfrominput.add_transition( s11, s11, 0.5 )
    hmm_leanfrominput.add_transition( s11, s22, 0.5 )
    hmm_leanfrominput.add_transition( s22, s11, 0.5 )
    hmm_leanfrominput.add_transition( s22, s22, 0.5 )
    hmm_leanfrominput.bake()

    hmm_leanfrominput.fit(X, stop_threshold=0.1, verbose=True, n_jobs=8 )

elif (z=='2'):
    X = hmm.sample(30000, False)  

    hmm_leanfrominput = HiddenMarkovModel.from_samples(DiscreteDistribution, 2, X=[X], state_names=['Fever', 'Healthy'])

print (hmm_leanfrominput)
# END #

# Linux Only Graph
if os.name == 'posix':
    plt.figure()
    plt.subplot(121)
    hmm.plot()
    plt.subplot(122)
    hmm_leanfrominput.plot()
    plt.show()

#sequences = [ numpy.array(list("DCDDCDCDCDCDCNDCNDCDNCNDNCDNCNDNCND")),
#	      numpy.array(list("CCCCNDNDNDNCDNCNDNCNCDNCDNCD")),
#	      numpy.array(list("CDCDCDNCDNCNDNCDNCNCNNNDNDNDDDD")) ]