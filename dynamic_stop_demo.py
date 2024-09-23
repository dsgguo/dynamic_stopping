# -*- coding: utf-8 -*-
#
# Authors: Duan Shunguo<dsg@tju.edu.cn>
# Date: 2024/9/1
# -*- coding: utf-8 -*-
# SSVEP Classification Demo

from collections import OrderedDict
import numpy as np
from scipy.signal import sosfiltfilt
from sklearn.metrics import balanced_accuracy_score
import sys
sys.path.append("D:\Duan_code\MetaBCI\MetaBCI")
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds, 
    generate_loo_indices, match_loo_indices)
from metabci.brainda.algorithms.decomposition import (
    FBTRCA, FBTDCA, FBSCCA, FBECCA, FBDSP,
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy.integrate import quad, trapz

dataset = Wang2016()
delay = 0.14 # seconds
channels = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
srate = 250 # Hz
duration = 0.5 # seconds
n_bands = 3
n_harmonics = 5
events = sorted(list(dataset.events.keys()))
freqs = [dataset.get_freq(event) for event in events]
phases = [dataset.get_phase(event) for event in events]

Yf = generate_cca_references(
    freqs, srate, duration, 
    phases=None, 
    n_harmonics=n_harmonics)

start_pnt = dataset.events[events[0]][1][0]
paradigm = SSVEP(
    srate=srate, 
    channels=channels, 
    intervals=[(start_pnt+delay, start_pnt+delay+duration+0.1)], # more seconds for TDCA 
    events=events)

wp = [[8*i, 90] for i in range(1, n_bands+1)]
ws = [[8*i-2, 95] for i in range(1, n_bands+1)]
filterbank = generate_filterbank(
    wp, ws, srate, order=4, rp=1)
filterweights = np.arange(1, len(filterbank)+1)**(-1.25) + 0.25

def data_hook(X, y, meta, caches):
    filterbank = generate_filterbank(
        [[8, 90]], [[6, 95]], srate, order=4, rp=1)
    X = sosfiltfilt(filterbank[0], X, axis=-1)
    return X, y, meta, caches

paradigm.register_data_hook(data_hook)

set_random_seeds(64)
l = 5
models = OrderedDict([
    # ('fbscca', FBSCCA(
    #         filterbank, filterweights=filterweights)),
    # ('fbecca', FBECCA(
    #         filterbank, filterweights=filterweights)),
    # ('fbdsp', FBDSP(
    #         filterbank, filterweights=filterweights)),
    ('fbtrca', FBTRCA(
            filterbank, filterweights=filterweights)),
    # ('fbtdca', FBTDCA(
    #         filterbank, l, n_components=8, 
    #         filterweights=filterweights)),
])

X, y, meta = paradigm.get_data(
    dataset,
    subjects=[1],
    return_concat=True,
    n_jobs=1,
    verbose=False)

loo_indices = generate_loo_indices(meta)
        
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
            print(rhos)
            # dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] / sum(rho_i[i]) for i in rho_i])
            dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] for i in rho_i])
            extracted_dm = self._extract_dm(pred_labels,Y_test,dm_i)
            
            for key in aggregated_dm:
                aggregated_dm[key].extend(extracted_dm[key])
        dm0 = aggregated_dm['correct']#正确的dm_i      
        dm1 = aggregated_dm['incorrect']#错误的dm_i
        print("dm0 max:",dm0[np.argmax(dm0)])
        kde0 = gaussian_kde(dm0)
        kde1 = gaussian_kde(dm1)
        prob = len(aggregated_dm['correct'])/(len(aggregated_dm['correct'])+
                                              len(aggregated_dm['incorrect']))
        
         # 将结果存储在模型字典中
        self.model_dict[duration] = {'kde0': kde0, 'kde1': kde1, 'prob': prob}
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
        
    def predict(self,testX,testY,duration):
        if duration in self.model_dict:
            kde0,kde1,prob = self._get_model(duration)
            rhos = self.estimator.transform(testX)
            
            rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
            # dm_i = np.array([rho_ i[i][np.argmax(rho_i[i])] / sum(rho_i[i]) for i in rho_i])
            dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] for i in rho_i])
            H0 = kde0(dm_i)
            H1 = kde1(dm_i)
            return kde0,kde1,prob
        else:
            raise ValueError(f"No model found for duration: {duration}")

    def decide(self,data,duration):
        if duration in self.model_dict:
            kde0,kde1,prob = self._get_model(duration)
            rhos = self.estimator.transform(data)
            dm_i = np.array(rhos[np.argmax(rhos)])
            p_H0,_ = quad(kde0,dm_i,dm_i+0.001)
            p_H1,_ = quad(kde1,dm_i,dm_i+0.001)
            p_thre = prob*p_H0/(prob*p_H0+(1-prob)*p_H1)
            
            if prob > p_thre or duration > 1:
                return True
            else:
                return False
        else:
            raise ValueError(f"No model found for duration: {duration}")
    
        
for model_name in models:
    if model_name == 'fbtdca':
        filterX, filterY = np.copy(X[..., :int(srate*duration)+l]), np.copy(y)
    else:
        filterX, filterY = np.copy(X[..., :int(srate*duration)]), np.copy(y)
    
    filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)

    n_loo = len(loo_indices[1][events[0]])#n_loo = 6
    loo_accs = []
    for k in range(n_loo):
        train_ind, validate_ind, test_ind = match_loo_indices(
            k, meta, loo_indices)
        train_ind = np.concatenate([train_ind, validate_ind])

        trainX, trainY = filterX[train_ind], filterY[train_ind]
        testX, testY = filterX[test_ind], filterY[test_ind]
        
        Ds = DynamicStop(models[model_name],trainX,trainY,duration)
        Ds.train(duration)
        # rate = Ds.validate(testX,duration)
        # print(rate)

        # x 轴数据
       
        # 对 KDE 曲线进行积分
       


    # # 绘制 KDE 曲线
    #     plt.plot(x0, kde0_values, label='kde0')
    #     plt.plot(x1, kde1_values, label='kde1')
    #     plt.xlabel('dm_i values')
    #     plt.ylabel('Density')
    #     plt.title('KDE of dm_i')
    #     plt.legend()
    #     plt.show()
#     model = clone(models[model_name]).fit(
#         trainX, trainY,
#         Yf=Yf
#     )
        
#     rhos = model.transform(testX)
    
#     pred_labels = model.predict(testX)

#     loo_accs.append(
#         balanced_accuracy_score(testY, pred_labels))
# print("Model:{} LOO Acc:{:.2f}".format(model_name, np.mean(loo_accs)))







