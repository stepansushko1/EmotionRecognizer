import numpy as np

class SupportVectorMachine:
    
    def __init__(self, C, kernel):
        self.C = C                         
        self.kernel = kernel          # <---
        self.weigth = None
        self.supportVectors = None
        
    def createGramMatrix(self, trainX, trainY):
        
        GramMatrix = []
        
        for i in range(len(trainX)):
            tempArr = []
            for j in range(len(trainX)):
                tempArr.append(GRBF(trainX[i], trainX[j]))
            
            GramMatrix.append(np.array(tempArr))
            tempArr = []
                
        GramMatrix = np.array(GramMatrix)
        labelsMatrix = trainY.reshape(-1, 1)
        final = GramMatrix * np.matmul(labelsMatrix, labelsMatrix.T)
        
        return final
        
            
        
    def fit(self, trainX, trainY):
        
        gramMatrix = self.createGramMatrix(trainX, trainY)
        
        return gramMatrix
        
        # Lagrange Dual Problem
        
        # Constaints of weigth
        
        # Optimizer
        
        
    def predict(self, testX):
        pass
    
    
def GRBF(x1, x2):
    diff = x1 - x2
    # return np.exp(-np.dot(diff, diff) * len(x1) / 2)
    # diff = x1-x2
    # print("diff",np.linalg.norm(diff)* 1/2*(1)**2)
    K = np.exp(-1 * ((np.square(np.linalg.norm(diff))) * 0.00001))
    return K