import sys
sys.path.append("D:\Duan_code\MetaBCI\MetaBCI")
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad

from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)


class DynamicStop:
    def __init__(self,decoder,train_data,train_label,Yf=None):
        self.decoder = decoder#FBTRCA
        self.data = train_data
        self.label = train_label
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
    
    def train(self,duration):
        self.estimator = self.decoder.fit(self.data,self.label,Yf=self.Yf)
        spliter = EnhancedLeaveOneGroupOut(return_validate=False)
        aggregated_dm = {'correct': [], 'incorrect': []}  # 初始化空字典
      
        for train_ind, test_ind in spliter.split(self.data, y=self.label):
            X_train, Y_train = np.copy(self.data[train_ind]), np.copy(self.label[train_ind])
            X_test, Y_test = np.copy(self.data[test_ind]), np.copy(self.label[test_ind])
            model = self.decoder.fit(X_train, Y_train, Yf=self.Yf ) 
            pred_labels = model.predict(X_test)
            rhos = model.transform(X_test)
            rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
            # dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] / sum(rho_i[i]) for i in rho_i])
            dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] for i in rho_i])
            extracted_dm = self._extract_dm(pred_labels,Y_test,dm_i)
            
            for key in aggregated_dm:
                aggregated_dm[key].extend(extracted_dm[key])
        dm0 = aggregated_dm['correct']#正确的dm_i      
        dm1 = aggregated_dm['incorrect']#错误的dm_i
        print(f"dm0 length: {len(dm0)}, dm1 length: {len(dm1)}")
        kde0 = gaussian_kde(dm0)
        kde1 = gaussian_kde(dm1)
        prob = len(aggregated_dm['correct'])/(len(aggregated_dm['correct'])+
                                              len(aggregated_dm['incorrect']))
        
         # 将结果存储在模型字典中
        self.model_dict[duration] = {'kde0': kde0, 'kde1': kde1, 'prob': prob}
        print(f"model_dict: {self.model_dict}")
        return kde0,kde1,prob,dm0,dm1
    
    def _get_model(self,duration):
        model_info = self.model_dict[duration]
        kde0 = model_info['kde0']
        kde1 = model_info['kde1']
        prob = model_info['prob']
        return kde0,kde1,prob
    
    def validate(self,testX,duration):
        if duration in self.model_dict:
            kde0,kde1,_ = self._get_model(duration)
            rhos = self.estimator.transform(testX)
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
            kde0,kde1,prob = self._get_model(duration)
            
            rhos = self.estimator.transform(data)
            rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
            dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] for i in rho_i])
            p_H0,_ = quad(kde0,dm_i,dm_i+0.001)
            p_H1,_ = quad(kde1,dm_i,dm_i+0.001)
            p_thre = prob*p_H0/(prob*p_H0+(1-prob)*p_H1)
            print(f"p_H0: {p_H0}, p_H1: {p_H1}, p_thre: {p_thre}, prob: {prob}")
            if prob > p_thre or duration > 1:
                return True
            else:
                return False
        else:
            raise ValueError(f"No model found for duration: {duration}")