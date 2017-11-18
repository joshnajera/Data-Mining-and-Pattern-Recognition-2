from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import sklearn
import numpy as np
import math

num_features = 0
unused_features = {0,1,2,3,4,5,6,12,13,14,15}

features = np.array([]) # Feature Names
data = [[]]     # Feature Data
label = np.array([])    # Labels    ->  0: Net Loss     1: Net Gain
stock_symbols = set()

X = np.array([[]])

with open("./Resources/dow_jones_index.data") as file:

    '''                     '''
    ''' Getting data set up '''
    '''                     '''

    # Get number of samples
    num_samples = sum(1 for _ in file) - 1
    file.seek(0)

    # First line has the name of each feature
    feature_line = file.readline()

    # Get those names and store them
    for i, f in enumerate(feature_line.strip().split(',')):
        if i in unused_features:
            continue
        features = np.append(features, f)

    num_features = features.size
    data = [['' for _ in range(num_features)] for _ in range(num_samples)]
    label = np.zeros( num_samples, dtype=int)
    tmp_label = np.zeros( num_samples )

    # Extract feature data
    # Toss last 4 values of each feature-- They are used to produce propper label
    # Label as ->  0: Net Loss      1: Net Gain
    for i, sample in enumerate(file):
        current_feature_number = 0
        for j, feature_value in enumerate(sample.strip().split(',')):

            # Tossing unused data
            if j in unused_features:
                # Create our label
                if j == 13:
                    tmp_label[i] = feature_value
                continue

            if j == 1:
                stock_symbols.add(feature_value)


            if feature_value == '':
                data[i][current_feature_number] = None
            else:
                data[i][current_feature_number] = feature_value.strip('$')

            current_feature_number = current_feature_number + 1
    

    # num_features = num_features - 1

    # # Group by stock symbol so we can impute
    # # Remove stock symbol
    # # Impute so we don't have NaN
    # all_subsets = []
    # imp = sklearn.preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)
    # for curr_sym in stock_symbols:
    #     data_subset = list(filter(lambda x: x[0] == curr_sym, data))
    #     data_subset = np.array(list(map(lambda x: x[1:], data_subset)))
    #     data_subset = imp.fit_transform(data_subset)
    #     # all_subsets.append(data_subset)
    #     for subset in data_subset:
    #         all_subsets.append(subset)

    # X = np.array([all_subsets[0]])
    # for i, subset in enumerate(all_subsets):
    #     if i == 0:
    #         continue
    #     X = np.append(X, [subset], axis=0)
    # # print(X)

    positive_count = 0
    negative_count = 0

    for i, entry in enumerate(tmp_label):
        if entry > 0:
            label[i] = 1
            positive_count = positive_count + 1
        else:
            label[i] = 0
            negative_count = negative_count + 1
    
    print("Positive count: {}".format(positive_count))
    print("Negative count: {}".format(negative_count))



imp = sklearn.preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)
X = imp.fit_transform(data)
X = sklearn.preprocessing.normalize(X, norm='l2', axis=0)


''' Splits  '''
score = 0
tests = 750
matrix_total=[[0,0],[0,0]]
# for i in range(tests):

loo = sklearn.model_selection.LeaveOneOut()
# loo.get_n_splits(X)
# for train_index, test_index in loo.split(X):
#     # X_train, X_test, y_train, y_test = train_test_split(X, label, test_size = 1)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = label[train_index], label[test_index]
#     clf = GaussianNB()
#     clf.fit(X_train, y_train)
#     prediction = clf.predict(X_test)
#     CF_Matrix = sklearn.metrics.confusion_matrix(y_test, prediction)
#     for i in range(len(CF_Matrix)):
#         for j in range(len(CF_Matrix[i])):
#             matrix_total[i][j] = matrix_total[i][j] + CF_Matrix[i][j]
#     score += clf.score(X_test, y_test)


# print(matrix_total[0])
# print(matrix_total[1])

# score /= (tests)

# print("Accuracy: {}".format(score))
# print("True Negative Rate: {}".format(matrix_total[0][0]/(matrix_total[0][0] + matrix_total[1][0])))


print("\n\nKNN")

# # TODO-- Sweep accros n = 5 => sqrt(n=750)
# loo.get_n_splits(X)
# score = 0
# for train_index, test_index in loo.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = label[train_index], label[test_index]

#     neigh = KNeighborsClassifier(n_neighbors=12)
#     neigh.fit(X_train, y_train)
#     prediction = neigh.predict(X_test)
#     score += sklearn.metrics.zero_one_loss(y_test, prediction)
# score /= 750
# print(score)



# print("\n\nSVC")
# # TODO-- Test with kernel linear, poly-3, poly-5
# loo.get_n_splits(X)
# score = 0
# for train_index, test_index in loo.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = label[train_index], label[test_index]
#     clf = SVC(kernel='poly', degree=5)
#     clf.fit(X_train, y_train)
#     prediction = clf.predict(X_test)
#     score += sklearn.metrics.zero_one_loss(y_test, prediction)
# score /= 750
# print(score)

# TODO-- Re-format everything to use train_test_split rather than LOOCV
# TODO-- Fix following KMeans
loo.get_n_splits(X)
score = 0
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = label[train_index], label[test_index]
    kmeans = KMeans(n_clusters=2, random_state=None)
    kmeans.fit(X_train)
    print(kmeans.labels_)
    prediction = kmeans.predict(X_test)
    score += sklearn.metrics.zero_one_loss(y_test, prediction)
score /= 750
print(score)

