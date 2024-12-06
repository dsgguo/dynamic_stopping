# -*- coding: utf-8 -*-
#
# Authors: Duan Shunguo<dsg@tju.edu.cn>
# Date: 2024/9/1
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import clone
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)


class LDA:
    def __init__(self,decoder):
        self.decoder = decoder#FBTRCA
        self.model_dict = {}  # 初始化模型字典

        
    def _extract_dm(self, pred_labels, Y_test, dm_i):
        extracted = {'correct': [], 'incorrect': []}
        for i, (pred, true) in enumerate(zip(pred_labels, Y_test)):
            if pred == true:
                extracted['correct'].append(dm_i[i])
            else:
                extracted['incorrect'].append(dm_i[i])
        return extracted
    
    def train(self,X,Y,duration,Yf=None):
        data = X
        label = Y
        yf = Yf
        spliter = EnhancedLeaveOneGroupOut(return_validate=False)
        aggregated_dm = {'correct': [], 'incorrect': []}  # 初始化空字典
        lda = LinearDiscriminantAnalysis()
        for train_ind, test_ind in spliter.split(data, y=label):
            X_train, Y_train = np.copy(data[train_ind]), np.copy(label[train_ind])
            X_test, Y_test = np.copy(data[test_ind]), np.copy(label[test_ind])
            model = clone(self.decoder).fit(X_train, Y_train, Yf=yf) 
            pred_labels = model.predict(X_test)
            rhos = model.transform(X_test)
            rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
            # dm_i = np.array([[np.partition(rho_i[i], -1)[-1]/sum(rho_i[i]), 
            #                   np.partition(rho_i[i], -2)[-2]/sum(rho_i[i])] for i in rho_i])
            dm_i = np.array([[1, 
                              np.partition(rho_i[i], -2)[-2]
                              /np.partition(rho_i[i], -1)[-1]] for i in rho_i])
            extracted_dm = self._extract_dm(pred_labels,Y_test,dm_i)
            for key in aggregated_dm:
                aggregated_dm[key].extend(extracted_dm[key])
        dm0 = aggregated_dm['correct']     
        dm1 = aggregated_dm['incorrect']
        train_L = np.concatenate((dm0,dm1),axis=0)
        labels_L = np.concatenate((np.ones(len(dm0)), np.zeros(len(dm1))), axis=0)
        print(f"dm0:{len(dm0)}, dm1:{len(dm1)}")
        
        model = lda.fit(train_L,labels_L)
        
        
        estimator = clone(self.decoder).fit(data,label,Yf=yf)
        self.model_dict[duration] = {'lda_model': model,"estimator":estimator}
        return model
    
    def _get_model(self,duration):
        model_info = self.model_dict[duration]
        lda_model = model_info['lda_model']
        estimator = model_info['estimator']
        return lda_model,estimator
    
    def validate(self,testX,duration):
        if duration in self.model_dict:
            lda_model,estimator = self._get_model(duration)
            rhos = estimator.transform(testX)
            rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
            dm_i = np.array([[1, 
                              np.partition(rho_i[i], -2)[-2]
                              /np.partition(rho_i[i], -1)[-1]] for i in rho_i])
            # print(f"dm_i, {dm_i}")
            L = lda_model.predict(dm_i)
            
            return L
    

    def decide(self,data,duration,t_max=1):
        if duration in self.model_dict:
            lda_model,estimator = self._get_model(duration)
            
            rhos = estimator.transform(data)
            label = estimator.predict(data)
            rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
            dm_i = np.array([[1, 
                              np.partition(rho_i[i], -2)[-2]/np.partition(rho_i[i], -1)[-1]] for i in rho_i])
            # print(f"decide_dm_i, {dm_i}")
            L = lda_model.predict(dm_i)
            if L ==1 or duration >= t_max :
                return True,label
            else:
                return False,label
        else:
            raise ValueError(f"No model found for duration: {duration}")
