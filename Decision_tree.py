import numpy as np
import Models_ult as mu
import Data_loader
import ModelMeasurement as mm


class DecisionTree(mu.model):
    def __init__(self) -> None:
        self.numFeats = 0
        self.num = 0
        self.decisions = []
        self.X = []
        self.Y = []
        self.checkFeats = []
    
    def calEntropy(self, p_list):
        
        entropy = 0.0
        for p in p_list:
            entropy += (-p*np.log2(p) )
        
        return entropy
    
    def make_tree(self, cls_idx, curr_x, curr_y):
        curr_x = curr_x[cls_idx]
        curr_y = curr_y[cls_idx]
        
        feats_count = len(curr_x[0])
        
        # calc total entropy
        total_count = len(curr_x)
        total_pos = np.sum(np.where(curr_y==1,1,0))
        
        # if node is pure, the node is leaf
        if total_pos == total_count : 
            return {"leaf":1}
        if total_pos == 0:
            return {"leaf":0}
        
        p_pos = total_pos/total_count
        p_neg = 1 - p_pos
        
        # if node is not pure, keep on make tree
        total_entropy = self.calEntropy([p_pos, p_neg])
        
        # find best feat with largest gain
        max_gain = 0.0
        best_feat_id = -1
        best_clsIdx0 = []
        best_clsIdx1 = []
        for i in range(feats_count):
            
            # skip features that is used before
            # if(self.checkFeats[i]==1): continue
            
            feats_data = curr_x[:,i]
            # classify data into two class WRT mean
            mean = np.mean(feats_data)
            cls0_idx = [feats_data <  mean] # store idx
            cls1_idx = [feats_data >= mean]
            
            # when curr x_set cannot be classified by curr feature, skip this feature
            if np.sum(cls0_idx)==0 or np.sum(cls1_idx)==0: continue

            classes = [cls0_idx,cls1_idx]
            
            # entropy
            gains = total_entropy
            for cls in classes:
                cls_count = np.sum(cls[0])
                
                if cls_count==0:continue # this class has no members
                
                pos_count = np.sum(np.where(curr_y[cls]==1, 1, 0))
                
                
                if pos_count == cls_count:
                    continue
                if pos_count == 0:
                    continue
                
                p_pos = pos_count/cls_count
                p_neg = 1-p_pos
                entropy = self.calEntropy([p_pos, p_neg])
                gains -= (cls_count/total_count)*entropy
            
            if max_gain < gains:
                max_gain = gains
                best_feat_id = i
                best_clsIdx0 = cls0_idx
                best_clsIdx1 = cls1_idx
            
        # when there is no feature to classify data    
        if best_feat_id==-1:
            currX_count = len(curr_x)
            best_clsIdx0 = np.ones(currX_count)[0:currX_count//2]
            best_clsIdx1 = np.ones(currX_count)[currX_count//2:]
        
        # this feat is marked as used
        # self.checkFeats[best_feat_id] = 1

        return  {
                    best_feat_id:
                        [
                            self.make_tree(best_clsIdx0, curr_x, curr_y), 
                            self.make_tree(best_clsIdx1, curr_x, curr_y),
                            mean
                        ]
                }
                
                
    
    def train(self, x_train, y_train):
        self.numFeats = len(x_train[0])
        self.num = len(x_train)
        self.X = x_train
        self.Y = y_train
        self.checkFeats = np.zeros(self.numFeats, np.uint8)
        
        cls_idx = np.arange(self.num)
        self.decisions = self.make_tree(cls_idx, x_train, y_train)
        
        print("training completed")
        
                
    def predict(self, x_test):
        
        curr_dict = self.decisions
        while len(curr_dict) != 0:
            curr_featId = list(curr_dict.keys())[0]
            
            if curr_featId == "leaf":
                # print(curr_dict[curr_featId])
                return curr_dict[curr_featId]
            
            [dict1, ditc2, mean] = curr_dict.get(curr_featId)
            
            if x_test[curr_featId] < mean:
                curr_dict = dict1
            else:
                curr_dict = ditc2
            
            
            
        


      
            
############# test part
test_set, train_set = Data_loader.load_CSVdata("data\Prostate_Cancer.csv")

## data preprocess
x_train = np.float32(train_set[:, 2:])
# normalize x data
x_train = (x_train - np.mean(x_train, axis=0)) / np.var(x_train, axis=0)

# standardlize labels
y_train = np.expand_dims(Data_loader.quantifyLabel(train_set[:, 1]), 1)

## init parameters
softMargin_coeff = 1.0
train_times = 100
## init model
decisionTree = DecisionTree()
## train model
decisionTree.train(x_train,y_train)


############ Evaluation
# normalize test data
x_test = np.float32(test_set[:, 2:])
x_test = (x_test - np.mean(x_test, axis=0)) / np.var(x_test, axis=0)
# standardlize labels
y_test = np.expand_dims(Data_loader.quantifyLabel(test_set[:, 1]), 1)
# decisionTree.predict(x_test[0])

## Model measurement
measure = mm.ModelMeasure()
evaluate_times = 50
mAP = measure.calMAP(decisionTree, evaluate_times,  x_train, y_train, x_test, y_test)

