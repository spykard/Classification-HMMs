import numpy as np

# neg_feature_train = neg_feature.reindex(train_indexes, copy=True, axis=0)
# neu_feature_train = neu_feature.reindex(train_indexes, copy=True, axis=0)
# pos_feature_train = pos_feature.reindex(train_indexes, copy=True, axis=0)
# neg_feature_test = neg_feature.reindex(test_indexes, copy=True, axis=0)
# neu_feature_test = neu_feature.reindex(test_indexes, copy=True, axis=0)
# pos_feature_test = pos_feature.reindex(test_indexes, copy=True, axis=0)

# alltogether = np.stack([neg, neu, pos])
# print(alltogether.shape)

# temp_2 = temp_2.toarray()
# neg = neg[:, None]
# print(temp_3.shape)
# #print(temp_2)
# print(neg.shape)    
# #print(neg)

a = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
b = np.array([100, 200])
b = b[:, None]
print("The shapes are:",a.shape, b.shape)

#print(np.column_stack((a, np.transpose(b))))
success = np.column_stack((a, b))  # or hstack
print(success)
print("The new shape is:",success.shape)
quit()
#xx = np.stack((temp_2, neg), axis = 1)   

#xx = np.append(temp_2, neg, axis = 0)

#print(np.stack([temp_2, np.transpose(alltogether)]).shape)
#print(hstack([temp_2, np.transpose(alltogether)]).toarray())

# Debug
# print(temp_3.shape)

clf = KNeighborsClassifier()
clf.fit(success, labels_train)

test_data_1 = vectorizer.transform(data_test)
test_data_2 = tfidf.transform(test_data_1)
test_data_2 = test_data_2.toarray()

alltogether = np.stack([neg, neu, pos])

success_test = np.column_stack((test_data_2, alltogether))

predicted = clf.predict(success_test)
