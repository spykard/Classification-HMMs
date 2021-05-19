    ## (Unsupervised) Train - Old Implementation
    hmm_leanfrominput = HiddenMarkovModel.from_samples(DiscreteDistribution, len(documentSentiments), X=data_train_transformed, n_jobs=n_jobs, verbose=False, name="Finegrained HMM")
    
    # Find out which which State number corresponds to which documentSentiment respectively
    ...
    ### (Unsupervised) Predict
    ...  
    #Print_Result_Metrics(labels_test.tolist(), predicted, targetnames, silent_enable_2, time_counter, 0, "HMM "+str(n)+"th Order Supervised")
    ###


    ## (Unsupervised) Train - Old Implementation
    hmm_leanfrominput = HiddenMarkovModel.from_samples(DiscreteDistribution, len(documentSentiments), X=data_train_transformed, n_jobs=n_jobs, verbose=False, name="Finegrained HMM")

    # Find out which which State number corresponds to pos/neg/neu respectively
    positiveState = list()
    negativeState = list()
    neutralState = list()
    print()
    for x in range(0, len(documentSentiments)):
        if silent_enable != 1:
            print("State", hmm_leanfrominput.states[x].name, hmm_leanfrominput.states[x].distribution.parameters)
        temp_dict = hmm_leanfrominput.states[x].distribution.parameters[0]
        positiveState.append(temp_dict["p" * n])
        negativeState.append(temp_dict["n" * n])
        neutralState.append(temp_dict["u" * n])
    positiveState = positiveState.index(max(positiveState))
    negativeState = negativeState.index(max(negativeState))
    neutralState = neutralState.index(max(neutralState))
    print("Pos Index is", positiveState, "Neg Index is", negativeState, "Neu Index is", neutralState)
    ###

    ### (Unsupervised) Predict
    predicted = list()
    for x in range(0, len(data_test_transformed)):
        try:
            predict = hmm_leanfrominput.predict(data_test_transformed[x], algorithm='viterbi')
        except ValueError as err:  # Prediction failed, predict randomly
            print("Prediction Failed:", err)
            predict = [randint(0, 2)]

        if predict[-1] == positiveState:  # I only care about the last Prediction
            predicted.append("pos")
        elif predict[-1] == negativeState:
            predicted.append("neg")
        else:
            predicted.append("neu")

        #predicted.append(hmm_leanfrominput.states[predict[-1]].name)

    #Print_Result_Metrics(labels_test.tolist(), predicted, targetnames, silent_enable_2, time_counter, 0, "HMM "+str(n)+"th Order Supervised")
    ###
