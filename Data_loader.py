import csv
import random

def load_CSVdata(dataPath):
    with open(dataPath, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]

    print(data[0])
    
    n = len(data) //3

    test_set = data[1:n]
    train_set = data[n:]

    random.shuffle(train_set)

    return test_set, train_set

