import numpy as np
import Models_ult as mu

class ModelMeasure():
    def __init__(self) -> None:
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.positive_smpl = 0
        self.negative_smpl = 0
        self.total_smpl = 0
        
    # the ratio of predicts completely
    def recall(self):
        return self.TP/(self.TP+self.FN)
    
    # the precision of positive predictions
    def precision(self):
        if self.TP+self.FP == 0: return 0
        
        return self.TP/(self.TP+self.FP)
    
    # the accurracy of predictions
    def accuracy(self):
        return (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)
    
    # receiver operator characteristic
    def ROC(self):
        TPR = self.TP/(self.TP+self.FN)
        FPR = self.FP/(self.FP+self.TN)
    
    # evaluate model 
    def evaluate(self, truthposotive, truthnegative, falsepositive, falsenegative):
        self.TP = truthposotive
        self.TN = truthnegative
        self.FP = falsepositive
        self.FN = falsenegative
        self.positive_smpl = self.TP + self.FN
        self.negative_smpl = self.TN + self.FP
        self.total_smpl = self.TP + self.TN + self.FP + self.FN
        print("FP:%d, FN:%d, TN:%d, TP:%d, pos count: %d, neg count: %d, total: %d" % (falsepositive, falsenegative, truthnegative, truthposotive, self.positive_smpl, self.negative_smpl, self.total_smpl))
    
        recall = self.recall()
        precision = self.precision()
        accuracy = self.accuracy()
        print("recall: %.3f, precision: %.3f, accuracy: %.3f" % (recall, precision, accuracy))
        return recall, precision, accuracy
    
    # evaluate mAP
    def calMAP(self, model, iter, x_train, y_train):
        iteration = iter
        evaluation_list = []
        AP = 0.0
        
        for i in range(iteration):
            print("------------------------------------it[%d]----------------------------------------" % (i))
                
            model.train(iteration=100)
            
            correct_acc = 0
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for x in x_train:
                predict_var = model.predict(x)
                gt = y_train[1]
                
                # prediction is Truth
                if gt == predict_var:
                    correct_acc+=1
                    if predict_var ==1: TP+=1 
                    else: TN+=1
                # prediction is False
                else:
                    if predict_var==1: FP+=1
                    else: FN+=1
                    
                # print("GT is %s, Prediction is %s" % (test_data[1], predict_var))

            # print("Accuracy: %.3f, M:%d, B:%d, correct: %d" % (correct_acc/len(test_set), m_acc, b_acc, correct_acc))
            recall, precision, accuracy = self.evaluate(TP, TN, FP, FN)
            evaluation_list.append([recall, precision, accuracy])
                
            print("------------------------------------end----------------------------------------")

        # calculate mAP
        mAP = AP / iteration
        
        print(">>> mAP: %.3f" % (mAP))
        
        return mAP