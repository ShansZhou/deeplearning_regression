import numpy as np
import Data_loader
import ModelMeasurement as mm

# same to linear regression, addtionally using gradient descent and cross
# entropy
class logisticRegression():
    def __init__(self, batchSize, epoch, learnRate):
        self.W = []
        self.B = 0.0
        self.batchsize = batchSize
        self.epoch = epoch
        self.lr = learnRate

    def train(self, x_train, y_train):
        batchsize = self.batchsize
        epoch = self.epoch
        lr = self.lr
        
        feats_count = len(x_train[0])
        self.W = np.random.normal(size=(1, feats_count))
        # self.W = np.zeros((1, feats_count))
        
        for epo in range(epoch):
            numOfdataset = len(x_train)
            cost = 0.0
            for i in range(numOfdataset//batchsize):
                # process a batch
                x_batch = x_train[i*batchsize:(i+1)*batchsize]

                # forward process
                linaregs = (1/batchsize)*np.sum((np.dot(self.W, np.transpose(x_batch)) + self.B), axis=0)
                linaregs = np.reshape(linaregs, (batchsize,1))
                y_predict = self.sigmoid(linaregs)
                
                # convert label string to float
                y_label = y_train[i*batchsize:(i+1)*batchsize]

                # cost
                cost = -(1/batchsize)*np.sum(y_label * np.log(y_predict+1e-10) + (1 - y_label) * np.log(1 - y_predict +1e-10))

                # calcuate the gradient parameters
                dW = (1 / batchsize) * np.dot(np.transpose(y_predict - y_label), x_batch)
                dB = (1 / batchsize) * (np.sum(y_predict - y_label))
                
                # updates
                self.W = self.W - (lr * dW)
                self.B = self.B - (lr * dB)

            # print("epoch %d, cost: %.5f" % (epo, cost))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, x_test):

        y = np.dot(self.W, x_test) + self.B

        # apply sigmoid
        y_predict = self.sigmoid(y)

        if y_predict>0.5: return 1 
        else: return 0


# test part
test_set, train_set = Data_loader.load_CSVdata("./data/Prostate_Cancer.csv")

## data preprocess
x_train = np.float32(train_set[:, 2:])
# normalize x data
x_train = (x_train - np.mean(x_train, axis=0)) / np.var(x_train, axis=0)

# standardlize labels
y_train = np.expand_dims(Data_loader.quantifyLabel(train_set[:, 1]), 1)


# init parameters
BatchSize = 10
epoch_times = 100
learn_rate = 0.001

## init model
logisticReg = logisticRegression(BatchSize,epoch_times,learn_rate)
# ## train model
# logisticReg.train(x_train, y_train)

############ Evaluation
# normalize test data
x_test = np.float32(test_set[:, 2:])
x_test = (x_test - np.mean(x_test, axis=0)) / np.var(x_test, axis=0)
# standardlize labels
y_test = np.expand_dims(Data_loader.quantifyLabel(test_set[:, 1]), 1)

## Model measurement
measure = mm.ModelMeasure()
iteration = 100
mAP = measure.calMAP(logisticReg, 100,  x_train, y_train, x_test, y_test)


