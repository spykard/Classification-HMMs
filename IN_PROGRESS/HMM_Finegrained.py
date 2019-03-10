""" 
Sentiment Analysis: (mainly Supervised) Text Classification using Hidden Markov Models
"""

import pandas as pd
import AdvancedHMM


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
                if temp[0] in ["neg", "neu", "pos", "mix"]:
                    sequences[count].append(temp[0])              

                data[count] += temp[1]

print("--\n--Processed", count+1, "documents", "\n--Dataset Name:", dataset_name)

df = pd.DataFrame({'Labels': labels, 'Data': data, 'Sequences': sequences})

# 2. Remove empty instances from DataFrame, actually affects accuracy
emptySequences = df.loc[df.loc[:,'Sequences'].map(len) < 1].index.values
df = df.drop(emptySequences, axis=0).reset_index(drop=True)  # reset_Index to make the row numbers be consecutive again

# 3. Print dataset information
print("--Dataset Info:\n", df.describe(include="all"), "\n\n", df.head(3), "\n\n", df.loc[:,'Labels'].value_counts(), "\n--\n", sep="")

# 4. Balance the Dataset by Undersampling
if False:
    set_label = "neu"
    set_desired = 75

    mask = df.loc[:,'Labels'] == set_label
    df_todo = df[mask]
    df_todo = df_todo.sample(n=set_desired, random_state=random_state)
    df = pd.concat([df[~mask], df_todo], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

''' CHECK HOW THIS AFFECTS PERFORMANCE '''
# 5. Shuffle the Dataset, it seems to be too perfectly ordered
if False:
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

# MAIN
if True:
    # create Model
    general_mixture_model_labels = AdvancedHMM.general_mixture_model_label_generator(df.loc[:,"Sequences"], df.loc[:,"Labels"])
    print(general_mixture_model_labels)
elif False:
    # create Model
    print("lel")