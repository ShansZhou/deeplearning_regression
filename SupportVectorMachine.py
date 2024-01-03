import numpy as np
import Data_loader
import ModelMeasurement as mm
import Models_ult as mu

class SVMachine(mu.model):
    def __init__(self, X_train, Y_train, soft_coef=1.0) -> None:
        
        # bias
        self.b = 0.0
        # soft margin coef
        self.C = soft_coef
        # alpha list
        self.alpha = []
        # error list wrt current alpha
        self.E = []
        
        # store (k dot k) in matrix
        self.productMat = []
        
        # format train data
        self.num = len(X_train)
        self.numFeats = len(X_train[0])
        self.X = X_train
        self.Y = Y_train
        
        
    def train(self, iteration=100):
        
        X = self.X
        Y = self.Y
        num = self.num
        
        # init alpha list
        self.alpha = np.ones((num,1))
        # init product matrix
        self.calProductMat()
        # init error list
        self.E = self.calcualteError()
        
        
                
        for iter in range(iteration):
            
            i, j = self.selectAlpha()
            
            if i == -1 and j == -1:
                print("i is -1 and j is -1, all alpha are valid")
                break
            
            # define bounds of alpha
            if Y[i] == Y[j]:
                L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                H = min(self.C, self.alpha[i] + self.alpha[j])
            else:
                L = max(0, self.alpha[j] - self.alpha[i])
                H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
            
            # calculate x with kernel for i,j
            k11 = self.kernel(X[i], X[i])
            k22 = self.kernel(X[j], X[j])
            k12 = self.kernel(X[i], X[j])
            
            eta = k11+k22 - 2*k12
            
            if eta <= 0: continue
            
            # update alpha
            # aj_new = aj_old + y_j*(E_i - E_j)/eta
            alpha_j_unclip = self.alpha[j] + (Y[j]*(self.E[i] - self.E[j]) ) / eta
            alpha_j = np.clip(alpha_j_unclip,L,H)
            # ai_new = ai_old + y_i*y_j(aj_new - aj_old)
            alpha_i = self.alpha[i] + Y[i]*Y[j]*(alpha_j - self.alpha[j])
            
            
            # update bais
            b_i = -self.E[i] + (self.alpha[i] - alpha_i)*Y[i]*k11 + \
                (self.alpha[j] - alpha_j)*Y[j]*k12 + self.b
            
            b_j = -self.E[j] + (self.alpha[i] - alpha_i)*Y[i]*k12 + \
                (self.alpha[j] - alpha_j)*Y[j]*k22 + self.b
            
            if 0 < alpha_i < self.C:
                b_new = b_i
            elif 0 < alpha_j < self.C:
                b_new = b_j
            else:
                b_new = (b_i+b_j)*0.5
            
            self.alpha[i] = alpha_i
            self.alpha[j] = alpha_j
            self.b = b_new

            # udpate all error
            self.calcualteError()
            
    def quantifyLabel(self, y_labels):
        return np.where(y_labels=="M", 1 ,0)
    
    # kernel for input X
    def kernel(self, x1, x2):
        return np.dot(x1,x2)
    
    # matrix product kernerl, trick for Sigma caculation of K_ij
    def calProductMat(self):
        self.productMat = np.zeros((self.num, self.num), np.float32)
        for i in range(self.num):
            for j in range(self.num):
                self.productMat[i][j] = self.productMat[j][i] = self.kernel(self.X[i],self.X[j])
          
            
    # calculate error list
    # E_i = SIGMA(alpha_i*Y_i*K_ij)
    def calcualteError(self):
        X = self.X
        Y = self.Y
        E = np.transpose(np.dot(np.transpose(self.alpha*Y), self.productMat) + self.b) - Y
        return E

    # select alpha i, j
    def selectAlpha(self):
        X = self.X
        Y = self.Y
        
        # arrange valid alpha to the front, leave invalid at the back.
        # impove efficiency
        index_list = [i for i in range(self.num) if 0 < self.alpha[i] < self.C]
        non_satisfy_list = [i for i in range(self.num) if i not in index_list]
        index_list.extend(non_satisfy_list)
        
        for i in index_list:
            if self.checkKTT(i): continue
            
            E_i = self.E[i]
            # find the large different E of j
            if E_i >= 0:
                j = np.argmin(self.E)
            else:
                j = np.argmax(self.E)
            
            # once find a pair of i,j
            return i, j

        return -1, -1
    
    # check KTT conditions
    # a_i = 0 -> y_i*f(x_i) >= 1
    # 0 < a_i < C -> y_i*f(x_i) == 1
    # a_i = C -> y_i*f(x_i) <= 1
    def checkKTT(self, i):
        X = self.X
        Y = self.Y
        a_i = self.alpha[i]
        
        y_fx = np.transpose(np.dot(np.transpose(self.alpha*Y), self.productMat[i])) + self.b
        y_fx = y_fx* Y[i]
        
        # if true, KKT is valied, otherwise KKT is invalid
        # if it is invalid, the alpha is going to update
        if a_i ==0:
            return y_fx >= 1 
        elif 0 < a_i <self.C:
            return y_fx == 1
        else:
            return y_fx <= 1
        
    def predict(self, x):
        X = self.X
        Y = self.Y
        r = self.b
        for i in range(len(self.alpha)):
            x_kernel = self.kernel(np.transpose(x), X[i])
            r += self.alpha[i]*Y[i]*x_kernel
            
        return 1 if r>0 else 0
        
            
############# test part
test_set, train_set = Data_loader.load_CSVdata("data\Prostate_Cancer.csv")

## data preprocess
x_train = np.float32(train_set[:, 2:])
# normalize x data
x_train = (x_train - np.mean(x_train, axis=0)) / np.var(x_train, axis=0)

# standardlize labels
y_train = np.expand_dims(Data_loader.quantifyLabel(train_set[:, 1]), 1)

## init model
svm = SVMachine(x_train, y_train, 1.0)

############ Evaluation
## Model measurement
measure = mm.ModelMeasure()
iteration = 100
mAP = measure.calMAP(svm, 100, x_train, y_train)


