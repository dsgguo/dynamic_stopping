# -*- coding: utf-8 -*-
#
# Authors: Duan Shunguo<dsg@tju.edu.cn>
# Date: 2024/9/1
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from scipy.signal import sosfiltfilt
from sklearn.metrics import balanced_accuracy_score
import sys
sys.path.append(r"D:\Duan_code\MetaBCI\MetaBCI")
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
from algorithm.Bayes import Bayes
from algorithm import analyze
dataset = Wang2016()
delay = 0.14 # seconds
channels = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
srate = 250 # Hz

n_bands = 3
n_harmonics = 5
events = sorted(list(dataset.events.keys()))
freqs = [dataset.get_freq(event) for event in events]
phases = [dataset.get_phase(event) for event in events]

start_pnt = dataset.events[events[0]][1][0]

paradigm = SSVEP(
    srate=srate, 
    channels=channels, 
    # intervals=[(start_pnt+delay, start_pnt+delay+duration+0.1)], # more seconds for TDCA 
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
        
Ds = Bayes(models['fbtrca'])
for duration in [0.5,0.6,0.7,0.8,0.9,1.0]:
    duration = duration 
    print(f"Train_Duration: {duration}")
    filterX, filterY = np.copy(X[..., int(srate*delay):int(srate*(delay+duration))]), np.copy(y)
    filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)
  
    train_ind, validate_ind, _ = match_loo_indices(
        5, meta, loo_indices)
    train_ind = np.concatenate([train_ind, validate_ind])

    trainX, trainY = filterX[train_ind], filterY[train_ind]
    Ds.train(trainX,trainY,duration)
        
    # kde0,kde1,prob,dm0,dm1 = Ds.train(trainX,trainY,duration)
    # p_H0_total,_ = quad(kde0,dm0[np.argmin(dm0)],dm0[np.argmax(dm0)])
    # p_H1_total,_ = quad(kde1,dm1[np.argmin(dm1)],dm1[np.argmax(dm1)])
    
tlabels = []
plabels = []
for i in range(0,40):
    filterX, filterY = np.copy(X[..., int(srate*delay):int(srate*(delay+5))]),np.copy(y)
    filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)
    _, _, test_ind = match_loo_indices(
                     5, meta, loo_indices)
    testX, testY = filterX[test_ind], filterY[test_ind]
    bufferX = testX[i:i+1,:,:]
    bufferY = testY[i:i+1]
    a = 0.5
    default_duration = round(a,2)
    trail = bufferX[:,:,0:int(srate*default_duration)]
    bool,label = Ds.decide(trail,default_duration)
    while not bool:
        print("长度不足,增加0.1s")
        a += 0.1
        default_duration = round(a,2)
        trail = bufferX[:,:,0:int(srate*default_duration)]
        bool,label = Ds.decide(trail,default_duration)
    
    tlabels.append(bufferY[0])
    plabels.append(label)
    print(f"Current duration: {default_duration}")    
     # 记录label
    print(f"Trail {i}: Label = {label}, Duration = {default_duration}")
acc = analyze.acc(plabels, tlabels)
itr = analyze.ITR(40, acc, 0.5)


