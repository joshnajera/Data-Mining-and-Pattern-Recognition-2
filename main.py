from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sklearn
import numpy as np

num_features = 0
unused_features = {0,1,2,12,13,14,15}

features = np.array([]) # Feature Names
data = np.array([])     # Feature Data
label = np.array([])    # Labels    ->  0: Net Loss     1: Net Gain


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
    data = np.zeros( (num_samples, num_features) )
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

            if feature_value == '':
                data[i][current_feature_number] = None
            else:
                data[i][current_feature_number] = feature_value.strip('$')

            current_feature_number = current_feature_number + 1

    for i, entry in enumerate(tmp_label):
        if entry > 0:
            label[i] = 1
        else:
            label[i] = 0






''' Splits  '''

imp = sklearn.preprocessing.Imputer(missing_values="NaN", strategy="mean",axis=0)
data = imp.fit_transform(data)

percent_right = 0
tests = 300
for i in range(tests):

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 100)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    for pred, act in zip(prediction, y_test):
        if pred == act:
            percent_right +=1

percent_right = percent_right / (tests*100)

print("Accuracy: {}".format(percent_right))
