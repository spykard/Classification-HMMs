''' 
Plot anything - such as Time Complexity of models - on a matplotlib graph, in order to perform comparisons
'''

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from collections import defaultdict

my_dictionary = {'HMM 1th Order Supervised': [0.4952049255371094, 2.422726631164551, 0.7479569911956787, 0.4513828754425049, 0.77882981300354], 
                'HMM 2th Order Supervised': [0.06107068061828613, 0.061005353927612305, 0.06319475173950195, 0.06075716018676758, 0.05847907066345215], 
                'HMM 3th Order Supervised': [0.06556582450866699, 0.06050372123718262, 0.06210613250732422, 0.06133723258972168, 0.058890342712402344], 
                'HMM 4th Order Supervised': [0.0648648738861084, 0.06398439407348633, 0.06446337699890137, 0.06129121780395508, 0.06092357635498047], 
                'HMM 5th Order Supervised': [0.06830716133117676, 0.06476235389709473, 0.06465864181518555, 0.06262588500976562, 0.06305217742919922], 
                'HMM 6th Order Supervised': [0.06929302215576172, 0.06514501571655273, 0.06669831275939941, 0.06406736373901367, 0.06318926811218262], 
                'HMM 7th Order Supervised': [0.06793856620788574, 0.06476020812988281, 0.06990766525268555, 0.06174135208129883, 0.05975198745727539], 
                'HMM 8th Order Supervised': [0.06361627578735352, 0.06474924087524414, 0.0643153190612793, 0.06054186820983887, 0.0595245361328125], 
                'HMM 9th Order Supervised': [0.06266951560974121, 0.062264204025268555, 0.06053781509399414, 0.05600714683532715, 0.057684898376464844], 
                'HMM 10th Order Supervised': [0.06163907051086426, 0.0604248046875, 0.060751914978027344, 0.05442047119140625, 0.05899214744567871], 
                'Ensemble of 1+2+3 Orders': [0.00047969818115234375, 0.0004611015319824219, 0.0007653236389160156, 0.0008823871612548828, 0.0005407333374023438]}
# my_dictionary = {'HMM 1th Order Supervised': [0, 0, 0, 0, 0, ], 
#                 'HMM 2th Order Supervised': [0, 0, 0, 0, 0, ], 
#                 'HMM 3th Order Supervised': [0, 0, 0, 2, 0, ], 
#                 'HMM 4th Order Supervised': [6, 11, 3, 10, 10, ], 
#                 'HMM 5th Order Supervised': [13, 21, 10, 16, 17, ], 
#                 'HMM 6th Order Supervised': [20, 23, 16, 17, 19, ], 
#                 'HMM 7th Order Supervised': [22, 20, 12, 20, 15, ], 
#                 'HMM 8th Order Supervised': [21, 18, 12, 20, 18, ], 
#                 'HMM 9th Order Supervised': [23, 16, 13, 21, 20, ], 
#                 'HMM 10th Order Supervised': [18, 11, 12, 20, 19, ]}
my_average = defaultdict(list) 
dataset_name = "Finegrained Sentiment Dataset"
k = 5

print(my_dictionary)

for model in my_dictionary:
    avg = np.mean(my_dictionary[model], axis=0)
    print(model, ": Average is", avg)  # "{:0.4f}".format(avg))
    my_average[model] = avg  # Save the average on a variable



print("Plotting AVERAGES of a List...")
indices = np.arange(len(my_average))
scores = []
model_names = []
for model in my_average:
    scores.append(my_average[model]) 
    model_names.append(model)
            
# Reverse the items to appear in correct order
scores.reverse()
model_names.reverse()

fig, ax1 = plt.subplots(figsize=(15, 8))
fig.subplots_adjust(left=0.18, top=0.92, bottom=0.08)
fig.canvas.set_window_title(dataset_name + " - Averages across " + str(k) + "-fold Cross Validation")

#p1 = ax1.barh(indices + 0.35, scores_acc, align="center", height=0.35, label="Accuracy (%)", color="cornflowerblue", tick_label=model_names)    
p2 = ax1.barh(indices, scores, align="center", height=0.35, label="Accuracy (%)", color="navy", tick_label=model_names)

ax1.set_title(dataset_name + " - Averages across " + str(k) + "-fold Cross Validation")
#ax1.set_xlim([0, 1])
ax1.xaxis.set_major_locator(MaxNLocator(11))
ax1.xaxis.grid(True, linestyle='--', which="major", color="grey", alpha=.25)
#ax1.legend((p1[0], p2[0]), ("Accuracy", "F1-score"))

# Right-hand Y-axis
# indices_new = []
# for i in range(0, len(model_names)):  # Trick to print text on the y axis for both bars
#     indices_new.append(indices[i])
#     indices_new.append(indices[i] + 0.35) 

ax2 = ax1.twinx()
ax2.set_yticks(indices)
ax2.set_ylim(ax1.get_ylim())  # Make sure that the limits are set equally on both yaxis so the ticks line up
#ax2.set_yticklabels(x for x in itertools.chain.from_iterable(itertools.zip_longest(scores_f1,scores_acc)) if x)  # Combine two lists in an alternating fashion
ax2.set_yticklabels(scores)
ax2.set_ylabel("Time (sec)")

plt.show()
print()