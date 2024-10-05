import sys
sys.path.append(r"d:\User_code\dsg\duanshunguo\MetaBCI\MetaBCI-master")
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)

class DummyKDE:
    def __init__(self,constant=0):
        self.dummy = DummyClassifier(strategy='constant', constant=constant)
        self.dummy.fit(np.zeros((1)),np.zeros(1))
    def __call__(self,x):
        X = np.array(x).reshape(-1,1)
        dummy_prob = self.dummy.predict(X)
        return dummy_prob


class Bayes:
    def __init__(self,decoder,Yf=None):
        self.decoder = decoder#FBTRCA
        self.model_dict = {}  # 初始化模型字典
        self.Yf = Yf
        
        
    def _extract_dm(self, pred_labels, Y_test, dm_i):
        # result = {'correct': [], 'incorrect': []}
        extracted = {'correct': [], 'incorrect': []}
        for i, (pred, true) in enumerate(zip(pred_labels, Y_test)):
            if pred == true:
                # result['correct'].append(i)
                extracted['correct'].append(dm_i[i])
            else:
                # result['incorrect'].append(i)
                extracted['incorrect'].append(dm_i[i])
        return extracted
    
    def train(self,X,Y,duration):
        data = X
        label = Y
        spliter = EnhancedLeaveOneGroupOut(return_validate=False)
        aggregated_dm = {'correct': [], 'incorrect': []}  # 初始化空字典
        prob_list = []
        for train_ind, test_ind in spliter.split(data, y=label):
            X_train, Y_train = np.copy(data[train_ind]), np.copy(label[train_ind])
            X_test, Y_test = np.copy(data[test_ind]), np.copy(label[test_ind])
            model = clone(self.decoder).fit(X_train, Y_train, Yf=self.Yf ) 
            pred_labels = model.predict(X_test)
            rhos = model.transform(X_test)
            rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
            # dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] / sum(rho_i[i]) for i in rho_i])
            dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] for i in rho_i])
            extracted_dm = self._extract_dm(pred_labels,Y_test,dm_i)
            sub_prob = len(extracted_dm['correct'])/(len(extracted_dm['correct'])+
                                              len(extracted_dm['incorrect']))
            prob_list.append(sub_prob)
            for key in aggregated_dm:
                aggregated_dm[key].extend(extracted_dm[key])
        dm0 = aggregated_dm['correct']#正确的dm_i      
        dm1 = aggregated_dm['incorrect']#错误的dm_i
        print(f"dm0 length: {len(dm0)}, dm1 length: {len(dm1)}")
        kde0 = gaussian_kde(dm0)
        if len(dm1) > 2:
            kde1 = gaussian_kde(dm1)
        else:
            kde1 = DummyKDE(0)
        prob = np.mean(prob_list)
        
         # 将结果存储在模型字典中
        estimator = clone(self.decoder).fit(data,label,Yf=self.Yf)
        self.model_dict[duration] = {'kde0': kde0, 'kde1': kde1, 'prob': prob,"estimator":estimator}
        return kde0,kde1,prob,dm0,dm1
    
    def _get_model(self,duration):
        model_info = self.model_dict[duration]
        kde0 = model_info['kde0']
        kde1 = model_info['kde1']
        prob = model_info['prob']
        estimator = model_info['estimator']
        return kde0,kde1,prob,estimator
    
    def validate(self,testX,duration):
        if duration in self.model_dict:
            kde0,kde1,_,estimator = self._get_model(duration)
            rhos = estimator.transform(testX)
            rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
            # dm_i = np.array([rho_ i[i][np.argmax(rho_i[i])] / sum(rho_i[i]) for i in rho_i])
            dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] for i in rho_i])
            H0 = kde0(dm_i)
            H1 = kde1(dm_i)
            H0_classified = []
            for i in range(len(dm_i)):
                if H0[i] > H1[i]:
                    H0_classified.append(dm_i[i])
            
            H0_ratio = len(H0_classified) / len(dm_i)
            return H0_ratio
        else:
            raise ValueError(f"No model found for duration: {duration}")

    def decide(self,data,duration):
        if duration in self.model_dict:
            kde0,kde1,prob,estimator = self._get_model(duration)
            
            rhos = estimator.transform(data)
            rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
            dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] for i in rho_i])
            p_H0 = kde0(dm_i)
            p_H1 = kde1(dm_i)
            p_pre = prob*p_H0/(prob*p_H0+(1-prob)*p_H1)
            p_thre = 0.95
            print(f"p_H0: {p_H0}, p_H1: {p_H1}, p_thre: {p_pre}, prob: {prob}")
            if p_pre >= p_thre or duration > 1 :
                return True
            else:
                return False
        else:
            raise ValueError(f"No model found for duration: {duration}")