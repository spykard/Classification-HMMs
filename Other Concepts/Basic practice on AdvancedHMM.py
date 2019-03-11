if __name__ == "__main__":
    if False:
        labels = [["dummy1", "pos", "neg", "neg", "dummy1"], ["dummy1", "pos"]]
        observations = [["dummy1", "good", "bad", "bad", "whateveromegalul"], ["dummy1", "good"]]
        golden_truth = ["pos", "neg"]
        labels_series = pd.Series(labels)
        observations_series = pd.Series(observations)
        golden_truth_series = pd.Series(golden_truth)
        hmm = AdvancedHMM()
        # General Settings
        # Data
        # Text Scenario
        # n-gram Settings
        # 1st Framework Training Settings
        # 1st Framework Prediction Settings

        hmm.build(architecture="A", model="State-emission HMM", framework="pome", k_fold=1, \
                state_labels_pandas=labels_series, observations_pandas=observations_series, golden_truth_pandas=golden_truth_series, \
                text_instead_of_sequences=[], text_enable=False,                            \
                n_grams=1, n_target="obs", n_prev_flag=False, n_dummy_flag=False,           \
                pome_algorithm="baum-welch", pome_verbose=False, pome_njobs=1,              \
                pome_algorithm_t="map"                                                      \
                )