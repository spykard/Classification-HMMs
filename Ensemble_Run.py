import pickle
import Ensemble_Framework

cross_val_prediction_matrix = []
mapping = []
golden_truth = []


cross_val_prediction_matrix.append(pickle.load(open('./Pickled Objects/temp1', 'rb')))
cross_val_prediction_matrix.append(pickle.load(open('./Pickled Objects/temp1_cl', 'rb')))
mapping.append(pickle.load(open('./Pickled Objects/temp2', 'rb')))
mapping.append(pickle.load(open('./Pickled Objects/temp2_cl', 'rb')))
golden_truth.append(pickle.load(open('./Pickled Objects/temp3', 'rb')))
golden_truth.append(pickle.load(open('./Pickled Objects/temp3_cl', 'rb')))

print(len(golden_truth[0][0]))
print(len(golden_truth[1][0]))

Ensemble_Framework.ensemble_run(cross_val_prediction_matrix, mapping, golden_truth, mode="sum", weights=[0.6, 0.4])