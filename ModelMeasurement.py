import numpy as np

class ModelMeasure():
    def __init__(self, truthposotive, truthnegative, falsepositive, falsenegative,) -> None:
        self.TP = truthposotive
        self.TN = truthnegative
        self.FP = falsepositive
        self.FN = falsenegative
        self.positive_smpl = self.TP + self.FN
        self.negative_smpl = self.TN + self.FP
        self.total_smpl = self.TP + self.TN + self.FP + self.FN
        print("FP:%d, FN:%d, TN:%d, TP:%d, pos count: %d, neg count: %d, total: %d" % (falsepositive, falsenegative, truthnegative, truthposotive, self.positive_smpl, self.negative_smpl, self.total_smpl))
    
    # the ratio of predicts completely
    def recall(self):
        return self.TP/(self.TP+self.FN)
    
    # the precision of positive predictions
    def precision(self):
        return self.TP/(self.TP+self.FP)
    
    # the accurracy of predictions
    def accuracy(self):
        return (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)
    
    # receiver operator characteristic
    def ROC(self):
        TPR = self.TP/(self.TP+self.FN)
        FPR = self.FP/(self.FP+self.TN)
    
    # evaluate model 
    def evaluate(self):
        recall = self.recall()
        precision = self.precision()
        accuracy = self.accuracy()
        print("recall: %.3f, precision: %.3f, accuracy: %.3f" % (recall, precision, accuracy))
        return recall, precision, accuracy
    
        
        