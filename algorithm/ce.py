# -*- coding: utf-8 -*-
#
# Authors: Duan Shunguo<dsg@tju.edu.cn>
# Date: 2024/9/1

import numpy as np
from sklearn.base import clone

class CE:
    def __init__(self,decoder,n_classes):
        self.decoder = decoder#FBTRCA
        self.model_dict = {}  # 初始化模型字典
        self.n_classes = n_classes
    
    def _cross_entropy(self, rho_i):
        n = self.n_classes
        rho_q = np.array([[np.partition(rho_i[i], -1)[-1], 
                              np.partition(rho_i[i], -2)[-2]] for i in rho_i])
        cost_h0 = np.sum(rho_i[0]) - n*np.log(np.sum(np.exp(rho_i[0])))
        cost_hq = np.array([rho_q[i, 0] - rho_q[i, 1] for i, _ in enumerate(rho_q)])
        return cost_h0, cost_hq
    
    def _get_model(self,duration):
        model_info = self.model_dict[duration]
        estimator = model_info['estimator']
        return estimator
    
    def train(self,X,Y,duration,Yf=None):
        data = X
        label = Y
        yf = Yf
        estimator = clone(self.decoder).fit(data,label,Yf=yf)
        self.model_dict[duration] = {"estimator":estimator}
        return estimator
    
    def decide(self,data,duration,t_max=1,thre=-1.5*1e-3):
        if duration in self.model_dict:
            estimator = self._get_model(duration)
            rhos = estimator.transform(data)
            label = estimator.predict(data)
            rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
            cost_h0,cost_hq = self._cross_entropy(rho_i)
            
            if -cost_h0*thre > -cost_hq or duration >= t_max :
                return True,label
            else:
                return False,label
        else:
            raise ValueError(f"No model found for duration: {duration}")