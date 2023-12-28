import numpy as np
import Data_loader

# solve a patch of data in one step without gradient descent
class linearRegression():
    def __init__(self):
        W = []
        B = 0.0

    def train(self, train_data):
        x = np.array(train_data) 
        y = np.array(train_data)

        x = np.float32(x[:,2:-1])
        y = np.float32(y[:,-1])

        mean_x = np.mean(x, axis=0)
        mean_y = np.mean(y, axis=0)

        x_var = x - mean_x
        y_var = np.expand_dims(y - mean_y, 1)
        xy_var = x_var * y_var
        xx_var = x_var**2

        # use global mean and variance to calculate prediction by one step.
        # besides, use garident descent for accurate WT and B
        self.W = np.sum(xy_var, axis=0) / np.sum(xx_var, axis=0)
        WT = np.transpose(np.expand_dims(self.W,1)) # WT*x + b = y
        self.B = mean_y - np.dot(WT, np.expand_dims(mean_x,1))

        print("W: ", self.W)
        print("B: ", self.B)

    def forward(self, x):
        
        WT = np.transpose(np.expand_dims(self.W,1))
        bias = np.expand_dims(self.B, 1)
        y = np.dot(WT, x) + bias

        return y


########### test part

test_set, train_set = Data_loader.load_CSVdata(dataPath="./data/Prostate_Cancer.csv")

linaReg = linearRegression()
linaReg.train(train_set)

test_data = np.float32(np.array(test_set)[:,2:-1]) 
print("input:", test_data[0,:])
x = np.expand_dims(test_data[0,:],1)
y_predict = linaReg.forward(x)
print("y_predict:", y_predict)