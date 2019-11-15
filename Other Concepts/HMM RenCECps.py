# Python 3

from pomegranate import *
from collections import defaultdict, OrderedDict, Counter
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ElementT
import os

# Load Dataset
count = 0
dataset = list()     # Training
observations = defaultdict(list)
datasetTest = list() # Testing
observationsTest = list() 
observationsTestSentenceCheckPoints = list()

path = "Samples of Ren-CECps 1.0/annotated xml"
for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    thefullpath = os.path.join(path, filename)
    tree = ElementT.parse(thefullpath)
    root = tree.getroot()

    print("\nCurrently processing Document #", count+1,"--", thefullpath)   

    currentDocument = list()
    for paragraph in root.findall('paragraph'):
        for sentence in paragraph.findall('sentence'):
            if (sentence.find('Joy').text == '0.0'):
                temp = '0'
            else:
                temp = '1'
            if (sentence.find('Hate').text == '0.0'):
                temp += '0'
            else:
                temp += '1'    
            if (sentence.find('Love').text == '0.0'):
                temp += '0'
            else:
                temp += '1' 
            if (sentence.find('Sorrow').text == '0.0'):
                temp += '0'
            else:
                temp += '1' 
            if (sentence.find('Anxiety').text == '0.0'):
                temp += '0'
            else:
                temp += '1' 
            if (sentence.find('Surprise').text == '0.0'):
                temp += '0'
            else:
                temp += '1'     
            if (sentence.find('Anger').text == '0.0'):
                temp += '0'
            else:
                temp += '1'      
            if (sentence.find('Expect').text == '0.0'):
                temp += '0'
            else:
                temp += '1'                                                                         
            currentDocument.append(temp)

            for keyword in sentence.findall('Keywords'):
                if (keyword.attrib['Joy'] == '0.0' or keyword.attrib['Joy'] == '0'):
                    temp2 = '0'
                else:
                    temp2 = '1'
                if (keyword.attrib['Hate'] == '0.0' or keyword.attrib['Hate'] == '0'):
                    temp2 += '0'
                else:
                    temp2 += '1'      
                if (keyword.attrib['Love'] == '0.0' or keyword.attrib['Sorrow'] == '0'):
                    temp2 += '0'
                else:
                    temp2 += '1'
                if (keyword.attrib['Sorrow'] == '0.0' or keyword.attrib['Sorrow'] == '0'):
                    temp2 += '0'
                else:
                    temp2 += '1'  
                if (keyword.attrib['Anxiety'] == '0.0' or keyword.attrib['Anxiety'] == '0'):
                    temp2 += '0'
                else:
                    temp2 += '1'
                if (keyword.attrib['Surprise'] == '0.0' or keyword.attrib['Surprise'] == '0'):
                    temp2 += '0'
                else:
                    temp2 += '1'  
                if (keyword.attrib['Anger'] == '0.0' or keyword.attrib['Anger'] == '0'):
                    temp2 += '0'
                else:
                    temp2 += '1'
                if (keyword.attrib['Expect'] == '0.0' or keyword.attrib['Expect'] == '0'):
                    temp2 += '0'
                else:
                    temp2 += '1'                                          

                if (count < 12): # Training
                    observations[temp].append(temp2) 
                else:            # Testing
                    observationsTest.append(temp2) 
            
            if (count >= 12):   # Testing
                observationsTestSentenceCheckPoints.append(len(observationsTest) - 1)

    if (count < 12):    
        dataset.append(currentDocument)
    else:
        datasetTest.append(currentDocument)
    count += 1
    # break

# (a) Transitions 
sentenceCodes = list()

for document in dataset:                          
    sentenceCodes.extend(document)

sentenceCodesUnique = list(OrderedDict.fromkeys(sentenceCodes)) # Count Total Number of Unique States (Sentence Codes)
print ("\nNumber of Unique States (Sentence Codes):",len(sentenceCodesUnique),"\n")

mapping = dict() # Create a Dictionary that maps State Names to the Matrix
for i, code in enumerate(sentenceCodesUnique):
    mapping[code] = i

n = len(sentenceCodesUnique)

matrixTransitions = np.zeros((n,n)) # Numpy for Sparsity

for document in dataset:    
    for (i,j) in zip(document,document[1:]):
        matrixTransitions[mapping[i]][mapping[j]] += 1

for row in matrixTransitions: # Convert to probabilities:
    s = float(sum(row))
    if (s > 0.0):
        row[:] = [f/s for f in row]

# (b) Observations Emitted
wordCodes = list()

for element in observations:                          
    wordCodes.extend(observations[element])

wordCodes.extend(observationsTest) # Include Test Set not just Training Set  

wordCodesUnique = list(OrderedDict.fromkeys(wordCodes)) # Count Total Number of Unique Word Codes
print ("Number of Unique Word Codes on Training + Test:",len(wordCodesUnique),"\n")     

# (c) Initial Distribution π
initials = Counter()
for document in dataset:
    initials[document[0]] += 1 # Beginning of Article    

#                    #
# HMM Implementation #
#                    #

# A list of all the States, s1,s2...s143
objectsStates = list()
for i in range(0, n):

    # (b)
    observationsFinal = Counter()
    for code in wordCodesUnique:
        observationsFinal[code] = 0

    totalObservations = float(len(observations[sentenceCodesUnique[i]]))
    if (totalObservations != 0.0):
        tempCount = Counter(observations[sentenceCodesUnique[i]])
        for element in tempCount:
            tempCount[element] = tempCount[element] / totalObservations        
        observationsFinal.update(tempCount)  # Create a combination of the Counter objects      
    else:  # Sentences with no Emotions
        for code in wordCodesUnique:
            observationsFinal[code] = 1/float(len(wordCodesUnique)) 

    d = DiscreteDistribution(dict(observationsFinal)) # Must not use the same d object for more than 1 State   
    objectsStates.append(State(d.copy(), name=sentenceCodesUnique[i]))

hmm = HiddenMarkovModel()

# objectsStates[0] represents sentenceCodesUnique[0]
for i in range(0, n): # X
    hmm.add_state(objectsStates[i])   
    for j in range(0, n): # Y
        if (matrixTransitions[i][j] > 0):                 
            hmm.add_transition(objectsStates[i], objectsStates[j], matrixTransitions[i][j])

# Initial Distribution π
for state in initials:
    hmm.add_transition(hmm.start, objectsStates[mapping[state]], initials[state]/float(len(dataset)))

hmm.bake()

#                    #
#       END          #
#                    #

print("Type 1 to Show detailed Stats")
x = input()

if (x == '1'):
    print (hmm)
    print (hmm.dense_transition_matrix())

# Linux Only Graph
if os.name == 'posix':
    hmm.plot()
    plt.show()

#                    #
#      PREDICT       #
#                    #

hmm_predictions = hmm.predict(observationsTest, algorithm='viterbi')

print("\nPredictions with Viterbi :")
for id in hmm_predictions:
    print (" -> ", end='')
    print (hmm.states[id].name, end='')

# print("\n\n", list(OrderedDict.fromkeys(observationsTestSentenceCheckPoints)))

# Lets Compare the Predictions with the Sentence Checkpoints
# sentence1 ((word *PREDICT* word *PREDICT* word *PREDICT* word *PREDICT*)), sentence2((word etc...
print("\n\nActual States :")
print (" -> None-start", end='')
count = 0
for i in range(0, len(hmm_predictions) - 1):
    if (i == list(OrderedDict.fromkeys(observationsTestSentenceCheckPoints))[count]):
        print (" -> ", end='')
        print (datasetTest[0][count], end='')
        count += 1
    else:
        print (" -> ........", end='')
print("\n")

#                    #
#       END          #
#                    #