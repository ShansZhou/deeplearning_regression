import numpy as np
import Data_loader
import ModelMeasurement as mm

# same to linear regression, addtionally using gradient descent and cross
# entropy
class logisticRegression():
    def __init__(self):
        self.W = []
        self.B = 0.0

    def train(self, train_set, batchsize=5, epoch=1000, lr=0.001):
        feats_count = len(train_set[0][2:])
        self.W = np.random.normal(size=(1, feats_count))
        # self.W = np.zeros((1, feats_count))
        
        for epo in range(epoch):
            numOfdataset = len(train_set)
            np.random.shuffle(train_set)
            cost = 0.0
            for i in range(numOfdataset//batchsize):
                # process a batch
                x_train = np.float32(train_set[i*batchsize:(i+1)*batchsize][:,2:])
                # normalize
                x_norm = (x_train - np.mean(x_train, axis=0)) / np.var(x_train, axis=0)
                # forward process
                linaregs = (1/batchsize)*np.sum((np.dot(self.W, np.transpose(x_norm)) + self.B), axis=0)
                y_predict = self.sigmoid(linaregs)
                
                # convert label string to float
                y_label = self.labelQuality(train_set[i*batchsize:(i+1)*batchsize][:,1])

                # cost
                cost = -(1/batchsize)*np.sum(y_label * np.log(y_predict+1e-10) + (1 - y_label) * np.log(1 - y_predict +1e-10))

                # calcuate the gradient parameters
                dW = (1 / batchsize) * np.dot(np.expand_dims(y_predict - y_label, axis=0), x_norm)
                dB = (1 / batchsize) * (np.sum(y_predict - y_label))
                
                # updates
                self.W = self.W - (lr * dW)
                self.B = self.B - (lr * dB)

            # print("epoch %d, cost: %.5f" % (epo, cost))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def labelQuality(self, label):
        return np.where(label=='M',1.0,0.0)
        
    def mapToLabel(self, y_predict):
        if y_predict > 0.5:
            return 'M'
        else:
            return 'B'

    def predict(self, x_test):

        y = np.dot(self.W, x_test) + self.B

        # apply sigmoid
        y_predict = self.sigmoid(y)

        return self.mapToLabel(y_predict)


# test part
test_set, train_set = Data_loader.load_CSVdata("./data/Prostate_Cancer.csv")
train_set = np.array(train_set)
test_set = np.array(test_set)

# init parameters
BatchSize = 10
epoch_times = 100
learn_rate = 0.001

logisticReg = logisticRegression()
logisticReg.train(train_set, batchsize=BatchSize, epoch=epoch_times, lr=learn_rate)

# Model measurement
test_all = np.float32(test_set[:][:,2:])
m_acc = np.sum(np.where(test_set[:][:,1]=="M",1,0))
b_acc = np.sum(np.where(test_set[:][:,1]=="B",1,0))

# Evaluation
iteration = 100
evaluation_list = []

for i in range(iteration):
    print("------------------------------------it[%d]----------------------------------------" % (i))
    
    logisticReg.train(train_set, batchsize=BatchSize, epoch=epoch_times, lr=learn_rate)
    
    correct_acc = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for test_data in test_set:
        x = np.expand_dims(np.float32(test_data[2:]),1)
        x_norm = (x - np.mean(test_all, axis=0).reshape(8,1)) / np.var(test_all, axis=0).reshape(8,1)
        
        predict_var = logisticReg.predict(x_norm)
        
        gt = test_data[1]
        
        # prediction is Truth
        if gt == predict_var:
            correct_acc+=1
            if predict_var =="M": TP+=1 
            else: TN+=1
        # prediction is False
        else:
            if predict_var=="M": FP+=1
            else: FN+=1
            
        # print("GT is %s, Prediction is %s" % (test_data[1], predict_var))

    # print("Accuracy: %.3f, M:%d, B:%d, correct: %d" % (correct_acc/len(test_set), m_acc, b_acc, correct_acc))
    
    modelmeasure = mm.ModelMeasure(TP, TN, FP, FN)
    recall, precision, accuracy = modelmeasure.evaluate()
    evaluation_list.append([recall, precision, accuracy])
    
print("------------------------------------end----------------------------------------")

# sort recall from low to high
evaluation_list_sorted = sorted(evaluation_list, key=lambda x:x[0], reverse=False)

# calculate AP
AP = 0.0
for i in range(iteration):
    
    recall = evaluation_list[i][0]
    precision = evaluation_list[i][1]
    
    if i==0: prev_recall=0
    else: prev_recall = evaluation_list[i-1][0]
    
    A = abs(recall - prev_recall) * precision
    # print("A[%d]: %.3f" % (i, A))
    AP += A

# calculate mAP
mAP = AP / iteration

print(">>> mAP: %.3f" % (mAP))


    

