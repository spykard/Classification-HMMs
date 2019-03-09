# Implementation of Example from https://www.mathworks.com/help/stats/hmmestimate.html

import matlab.engine

trans = matlab.double([[0.95,0.05], [0.10,0.90]])
emis = matlab.double([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [1/10, 1/10, 1/10, 1/10, 1/10, 1/2]])

print(trans.size, emis.size)

eng = matlab.engine.start_matlab()
output = eng.hmmgenerate(1000,trans,emis,nargout=2)

output_2 = eng.hmmestimate(output[0],output[1],nargout=2)

print(output_2)


# Implementation from Weird Labeled training Bug.py

# labels = [["dummy1", "pos", "neg", "neg", "dummy1"], ["dummy1", "pos"]]
# observations = [["dummy1", "good", "bad", "bad", "whateveromegalul"], ["dummy1", "good"]]
# # 5 is used to fill/pad the remaining spots, since Matlab doesn't accept different lengths
# labels_mapping = matlab.int32([[1, 2, 3], [1, 2, 3]])
# observations_mapping = matlab.int32([[1, 2, 3], [4, 5, 6]])

# output_3 = eng.hmmestimate(observations_mapping,labels_mapping,nargout=2)

# print(output_3[0])
# print(output_3[1])