# Reference: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
from random import seed
from random import randrange
from csv import reader

# Load CSV file to a dataset in form of a list
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Calculate the Minkowski distance between two vectors
def p_norm_distance(row1, row2, p):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += abs(row1[i] - row2[i]) ** p
    return distance ** (1/p)

# Calculate the Jaccard distance between two vectors
def jaccard_distance(row1, row2):
    numerator = 0
    x_times_y = 0
    x_sq = 0
    y_sq = 0
    for i in range(len(row1) - 1):
        numerator += (row1[i] - row2[i]) ** 2
        x_sq += row1[i] ** 2
        y_sq += row2[i] ** 2
        x_times_y += row1[i] * row2[i]
    return numerator / (x_sq + y_sq - x_times_y)

# Normalize dataset
def data_normalizer(data):
    max_list = list()
    min_list = list()
    for i in range(len(data[0])):
        col = [row[i] for row in data]
        min_list.append(min(col))
        max_list.append(max(col))
    for row in data:
        for col in range(len(data[0])):
            row[i] = (row[i] - min_list[i])/(max_list[i] - min_list[i])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Get accuracy of model
def get_accuracy(y, y_hat):
    total = 0
    for i in range(len(y)):
        if y_hat[i] == y[i]: total += 1
    return (float(total) / len(y)) * 100

# Evaluate an algorithm using a cross validation split
def get_scores(dataset, n_folds, num_neighbors, distance_method, *args):
    # Split dataset into n_folds folds
    folds = cross_validation_split(dataset, n_folds)
    # Calculate accuracies with cross validation
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        y_hat = k_nearest_neighbors(train_set, test_set, num_neighbors, distance_method, *args)
        y = [row[-1] for row in fold]
        accuracy = get_accuracy(y, y_hat)
        scores.append(accuracy)
    return scores

# Make a prediction with neighbors- dist_method is either p_norm_distance or jaccard_distance
def predict(train, test_row, num_neighbors, dist_method, *args):
    # Calculate distances of all train samples to the test sample
    distances = list()
    for train_row in train:
        dist = dist_method(test_row, train_row, *args)  # for p_norm_distance, *arg is "p"
        distances.append((train_row, dist))

    # Sort the distances
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()

    # Get the closest num_neighbors of neighbors
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    # Get the labels of the closest neighbors and find the major label
    neighbors_labels = [row[-1] for row in neighbors]
    predicted_label = max(set(neighbors_labels), key=neighbors_labels.count)
    return predicted_label

# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors, dist_method, *args):
    predictions = []
    for row in test:
        output = predict(train, row, num_neighbors, dist_method, *args)
        predictions.append(output)
    return predictions


# Test the kNN on the Iris Flowers dataset
dist_type = jaccard_distance
seed(1)
filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)
# evaluate algorithm
n_folds = 5
num_neighbors = 9
scores = get_scores(dataset, n_folds, num_neighbors, dist_type)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
