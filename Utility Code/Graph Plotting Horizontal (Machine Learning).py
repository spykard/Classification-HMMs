def Plot_Results(k, dataset_name):
    '''    Plot the Accuracy of all Classifiers in a Graph    '''
    global cross_validation_all, cross_validation_average

    print("Plotting AVERAGES of Cross Validation...")
    for model in cross_validation_all:
        avg = tuple(np.mean(cross_validation_all[model], axis=0))
        cross_validation_average[model] = avg  # Save the average on a global variable
    indices = np.arange(len(cross_validation_average))
    scores_acc = []
    scores_f1 = []
    model_names = []
    for model in cross_validation_average:
        scores_acc.append(cross_validation_average[model][0]) 
        scores_f1.append(cross_validation_average[model][1])
        model_names.append(model[1:-1])  # Remove Parentheses

    # Reverse the items to appear in correct order
    scores_acc.reverse()
    scores_f1.reverse()
    model_names.reverse()

    fig, ax1 = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(left=0.18, top=0.92, bottom=0.08)
    fig.canvas.set_window_title(dataset_name + " - Averages across " + str(k) + "-fold Cross Validation")
    
    p1 = ax1.barh(indices + 0.35, scores_acc, align="center", height=0.35, label="Accuracy (%)", color="navy", tick_label=model_names)    
    p2 = ax1.barh(indices, scores_f1, align="center", height=0.35, label="Accuracy (%)", color="cornflowerblue", tick_label=model_names)
   
    ax1.set_title(dataset_name + " - Averages across " + str(k) + "-fold Cross Validation")
    ax1.set_xlim([0, 1])
    ax1.xaxis.set_major_locator(MaxNLocator(11))
    ax1.xaxis.grid(True, linestyle='--', which="major", color="grey", alpha=.25)
    ax1.legend((p1[0], p2[0]), ("Accuracy", "F1-score"))

    # Right-hand Y-axis
    indices_new = []
    for i in range(0, len(model_names)):  # Trick to print text on the y axis for both bars
        indices_new.append(indices[i])
        indices_new.append(indices[i] + 0.35) 

    ax2 = ax1.twinx()
    ax2.set_yticks(indices_new)
    ax2.set_ylim(ax1.get_ylim())  # Make sure that the limits are set equally on both yaxis so the ticks line up
    ax2.set_yticklabels(x for x in itertools.chain.from_iterable(itertools.zip_longest(scores_f1,scores_acc)) if x)  # Combine two lists in an alternating fashion
    ax2.set_ylabel("Performance")

    plt.show()
    print()
