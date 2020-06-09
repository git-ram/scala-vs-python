from random import randrange
from csv import reader
from itertools import repeat
import concurrent.futures
import time, csv, os

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

# Convert string column to float for numerical values
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
    return dataset

# Convert string column to integer for categorical values
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return dataset

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Normalize dataset
def data_normalizer(data):
    max_list = list()
    min_list = list()
    for i in range(len(data[0]) - 1):
        col = [row[i] for row in data]
        min_list.append(min(col))
        max_list.append(max(col))
    for row in data:
        for col in range(len(data[0]) - 1):
            row[col] = (row[col] - min_list[col]) / (max_list[col] - min_list[col])
    return data


def data_prep(dataset):
    for col in range(len(dataset[0])):
        if is_float(dataset[0][col]):
            dataset = str_column_to_float(dataset, col)
        else:
            dataset = str_column_to_int(dataset, col)

    dataset = data_normalizer(dataset)
    return dataset

# Calculate the Minkowski distance between two vectors
def p_norm_distance(row1, row2, p):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += abs(row1[i] - row2[i]) ** p
    return distance ** (1 / p)

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

# Evaluate an algorithm using a given list of folds
def get_scores(f_list, num_neighbors, distance_method, *args):
    # Calculate accuracies with cross validation
    scores = list()
    for fold in f_list:
        train_set = list(f_list)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            #row_copy = list(row)
            test_set.append(row)
            #row_copy[-1] = None
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

# Compute scores via Jaccard and Minkowski for a particular number of neighbors
def compute(folds_list, num_folds, num_neighbor, p_norm_max):
    result_jaccard = {"#folds": [], "#neighbors": [], "mean accuracy": []}
    result_minkowski = {"#folds": [], "#neighbors": [], "p": [], "mean accuracy": []}

    # Jaccard method
    scores = get_scores(folds_list, num_neighbor, jaccard_distance)
    avg_score = sum(scores) / float(len(scores))
    # Store the results
    result_jaccard["#folds"].append(num_folds)
    result_jaccard["#neighbors"].append(num_neighbor)
    result_jaccard["mean accuracy"].append(avg_score)
    # print('Scores: %s' % scores)
    # print('num_folds: %d , num_neighbors: %d, dist_method: %s, mean Accuracy: %5.3f' %
    #      (num_folds, num_neighbor, 'Jaccard', avg_score))

    # p_norm_distance for various p values
    for p in range(1, p_norm_max + 1):
        scores = get_scores(folds_list, num_neighbor, p_norm_distance, p)
        avg_score = sum(scores) / float(len(scores))

        # Store the results
        result_minkowski["#folds"].append(num_folds)
        result_minkowski["#neighbors"].append(num_neighbor)
        result_minkowski["p"].append(p)
        result_minkowski["mean accuracy"].append(avg_score)

        # print('Scores: %s' % scores)
        # print('num_folds: %d , num_neighbors: %d, dist_method: %s, p: %d, mean Accuracy: %5.3f' %
        #      (num_folds, num_neighbor, 'Minkowski', p, avg_score))

    #print("Computation done.")
    return [result_jaccard, result_minkowski]

def runner(filename, n_folds, num_neighbors_max, p_norm_max, parallel='seq'):
    p_norm_range = range(1, p_norm_max + 1)
    dataset = load_csv(filename)
    dataset = data_prep(dataset)

    # Split dataset into n_folds folds
    folds_list = cross_validation_split(dataset, n_folds)

    # Evaluate
    if parallel == 'mp':
        # print("*** Using multiprocessing ***")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            number_of_neighbors = range(1, num_neighbors_max + 1)

            results = executor.map(compute, repeat(folds_list), repeat(n_folds),
                                   number_of_neighbors, repeat(p_norm_max))

            # for result in results:
            # print(result)
    elif parallel == 'mt':
        # print("*** Using multithreading ***")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            number_of_neighbors = range(1, num_neighbors_max + 1)

            results = executor.map(compute, repeat(folds_list), repeat(n_folds),
                                   number_of_neighbors, repeat(p_norm_max))

    else:
        # print("*** Using sequential computation ***")
        result_jaccard = {"#folds": [], "#neighbors": [], "mean accuracy": []}
        result_minkowski = {"#folds": [], "#neighbors": [], "p": [], "mean accuracy": []}
        for num_neighbor in range(1, num_neighbors_max + 1):
            # Jaccard method
            scores = get_scores(folds_list, num_neighbor, jaccard_distance)
            avg_score = sum(scores) / float(len(scores))

            # Store the results
            result_jaccard["#folds"].append(n_folds)
            result_jaccard["#neighbors"].append(num_neighbor)
            result_jaccard["mean accuracy"].append(avg_score)
            # print('Scores: %s' % scores)
            # print('num_folds: %d , num_neighbors: %d, dist_method: %s, mean Accuracy: %5.3f' %
            #      (n_folds, num_neighbor, 'Jaccard', avg_score))

            # p_norm_distance
            for p in p_norm_range:
                scores = get_scores(folds_list, num_neighbor, p_norm_distance, p)
                avg_score = sum(scores) / float(len(scores))

                # Store the results
                result_minkowski["#folds"].append(n_folds)
                result_minkowski["#neighbors"].append(num_neighbor)
                result_minkowski["p"].append(p)
                result_minkowski["mean accuracy"].append(avg_score)

                # print('Scores: %s' % scores)
                #print('num_folds: %d , num_neighbors: %d, dist_method: %s, p: %d, mean Accuracy: %5.3f' %
                #      (n_folds, num_neighbor, 'Minkowski', p, avg_score))

NUM_TRIALS = 10
DATASET_LIST = ['iris.csv']
NUM_FOLD = 5
NUM_NEIGHBOR_MAX = 10    # This is eequivalent to number of processes that are used to calculate stuff
                         # Default is 10 which is bigger than 8 cores available
P_NORM_MAX = 6          # The numbeof of types of Minkowski distance calculations
COMP_METHOD = 'mt'      # The method of computation, multiprocess = 'mp', multithread = 'mt', sequential = 'seq'
print("Available processors: ", os.cpu_count())

def experiment_runner(filename_list, n_trials, parallel):

    for data in filename_list:
        avg_time_for_each_pnorm = {}
        for n_n in range(1, NUM_NEIGHBOR_MAX + 1):
                for p_n in range(1, P_NORM_MAX + 1):
                    trials_time_sum = 0
                    for i in range(n_trials):
                        start = time.perf_counter()
                        runner(filename=data, n_folds=NUM_FOLD, num_neighbors_max=n_n,
                               p_norm_max=p_n, parallel=COMP_METHOD)
                        finish = time.perf_counter()
                        trials_time_sum += round(finish - start, 4)
                    avg_time_for_each_pnorm[f"dataset {data} - num_neigh_max {n_n} - p_norm_max {p_n}"] = \
                                                                                    trials_time_sum/float(n_trials)
        # Write the results for each dataset into its own report file
        w = csv.writer(open(f"./Reports/Python_{data}_results_{COMP_METHOD}.csv", "w"))
        for k, v in avg_time_for_each_pnorm.items():
            w.writerow([k, v])

# Run the experiment
experiment_runner(filename_list=DATASET_LIST, n_trials=NUM_TRIALS, parallel=COMP_METHOD)
