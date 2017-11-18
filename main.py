from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import numpy as np
import sklearn
import math

num_features, num_samples = 0, 0
num_classes = 2

unused_features = {0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15}

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

    positive_count = 0
    negative_count = 0

    for i, entry in enumerate(tmp_label):
        if entry > 0:
            label[i] = 1
            positive_count = positive_count + 1
        else:
            label[i] = 0
            negative_count = negative_count + 1
    
    print("Positive samples: {}".format(positive_count))
    print("Negative samples: {}".format(negative_count))



imp = sklearn.preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)
X = imp.fit_transform(data)
X = sklearn.preprocessing.normalize(X, norm='l2', axis=0)


matrix_total=[[0,0],[0,0]]


def main():
    '''  Evaluates Gaussian Naive Bayes, KNN, SVC, and KMeans classifiers  '''

    gnb_score, knn_score, svc_score, kmeans_score = 0, 0, 0, 0
    num_tests = 100
    t_size = 1

    X_train, X_test, y_train, y_test = \
        train_test_split(X, label, test_size=t_size)

    print(gnb(X_train, y_train, X_test, y_test))
    print(knn(X_train, y_train, X_test, y_test))
    print(svm(X_train, y_train, X_test, y_test))


def gnb(X_train, y_train, X_test, y_test):
    ''' Runs Guassian Naive Bayes
    Input: X, y, test_size
    Return: score, confusion_matrix
    '''

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    score = 1 if (prediction == y_test) else 0
    # TODO-- Convert to ROC Curve sklearn.metrics.roc_curve 
    CF_Matrix = sklearn.metrics.confusion_matrix(y_test, prediction)

    return (score, CF_Matrix)

    # -- Refactor
    # for i in range(len(CF_Matrix)):
    #     for j in range(len(CF_Matrix[i])):
    #         matrix_total[i][j] = matrix_total[i][j] + CF_Matrix[i][j]
    # print("True Negative Rate: {}".format(matrix_total[0][0]/(matrix_total[0][0] + matrix_total[1][0])))



def knn(X_train, y_train, X_test, y_test):
    ''' Runs K-Nearest Neighbors
    Input: X, y, test_size
    Return: Dict of {n:score}
    '''

    results = dict()
    for neighbs in range(5, math.ceil(math.sqrt(num_samples))):
        clf = KNeighborsClassifier(n_neighbors=neighbs)
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        score = sklearn.metrics.zero_one_loss(y_test, prediction)
        results[neighbs] = score

    return results


def svm(X_train, y_train, X_test, y_test):
    ''' Runs Suport Vector Classification
    Input: X, y, test_size
    Return: all_results=(kernel_name[& degree], result)
    '''

    all_kernels, all_results = [], []
    all_kernels.append(('linear', SVC(kernel='linear')))
    all_kernels.append(('poly-d3', SVC(kernel='poly',degree=3)))
    all_kernels.append(('poly-d5', SVC(kernel='poly',degree=5)))
    
    for name, clf in all_kernels:
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        zero_one = sklearn.metrics.zero_one_loss(y_test, prediction)
        all_results.append((name, zero_one))

    return all_results


# TODO-- Fix KMeans
def kmeans(X_train, y_train, X_test, y_test):
    ''' Runs K-Means Clustering
    Input: X, y, test_size
    Return: result
    '''

    clf = KMeans(n_clusters=num_classes, random_state=None)
    clf.fit(X_train)
    prediction = clf.predict(X_test)
    result = sklearn.metrics.zero_one_loss(y_test, prediction)

    return result


if __name__ == '__main__':
    main()





        ##              Experiment               ##
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
        ##                                          ##
