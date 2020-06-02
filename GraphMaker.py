import numpy as np
import csv
import matplotlib.pyplot as plt

file_list = ['Python_abalone_multiP.csv', 'Python_adult_500_multiP.csv', 'Python_iris_multiP.csv',
             'Scala_abalone.csv', 'Scala_adult_500.csv', 'Scala_iris.csv']

def graph(files):
    for file in files:
        file_name = "./AllReports/" + file
        max_n_list = []
        time_list = []
        with open(file_name) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            count = 1
            for row in readCSV:
                if count % 6 == 0: # Read every 6th line i.e. when p = 6
                    max_neighbor = int((row[0].split("-"))[1].split(" ")[2])
                    time = float(row[1])
                    max_n_list.append(max_neighbor)
                    time_list.append(time)
                    count += 1
                else:
                    count += 1
                    continue
        from matplotlib.pyplot import figure
        figure(num=None, figsize=(8, 6), dpi=80)
        plt.title(file)

        plt.ylabel("Time (s)")
        plt.xlabel("Max Number of Neighbors")
        plt.scatter(max_n_list, time_list, color='red')
        plt.plot(max_n_list, time_list, color='blue')
        plt.xticks(np.arange(0, 11, 1))
        plt.yticks(np.arange(0, 50, 2))
        plt.savefig("./AllReports/" + file.split(".")[0])
        plt.show()


graph(file_list)

