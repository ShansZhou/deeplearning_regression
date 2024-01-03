import csv
import numpy as np


def load_CSVdata(dataPath):
    with open(dataPath, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]

    print(data[0])
    
    n = len(data) //3

    test_set = np.array(data[1:n]) 
    train_set = np.array(data[n:])

    return test_set, train_set

def quantifyLabel( y_labels):
        return np.where(y_labels=="M", 1 ,0)

def mapToLabel(y_predict):
    if y_predict > 0.5:
        return 'M'
    else:
        return 'B'