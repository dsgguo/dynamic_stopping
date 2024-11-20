# -*- coding: utf-8 -*-
#
# Authors: Duan Shunguo<dsg@tju.edu.cn>
# Date: 2024/9/1

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
import joblib

class DummyKDE:
    """
    A dummy KDE class that uses a DummyClassifier to come up with insufficent negative samples .

    Attributes:
        dummy (DummyClassifier): A dummy classifier with a constant strategy.
    """
    def __init__(self,constant=0):
        self.dummy = DummyClassifier(strategy='constant', constant=constant)
        self.dummy.fit(np.zeros((1)),np.zeros(1))
    def __call__(self,x):
        X = np.array(x).reshape(-1,1)
        dummy_prob = self.dummy.predict(X)
        return dummy_prob


class Bayes:
    """
    The Bayes_based Dynamic Stopping Algorithm for handling Bayesian decoding.

    Attributes:
        decoder: The decoder for EEG to be used.
        model_dict (dict): A dictionary to store Estimator, KDE_models and the Prior possibility.
    """
    def __init__(self,decoder,t_max=1,srate=1000,user_mode=0):
        self.decoder = decoder
        self.model_dict = {}
        self.t_max = t_max
        self.user_mode = user_mode
        self.srate = srate 

    def _save_model(self, filename):
        """
        Saves the model to a file.

        Args:
            filename (str): File name.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        joblib.dump(self.model_dict, filename)
    
    def _load_model(self, filename):
        """
        Loads the model from a file.

        Args:
            filename (str): File name.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        self.model_dict = joblib.load(filename)
            
    def _extract_dm(self, pred_labels, Y_test, dm_i):
        """
        Extracts decision metrics from predicted and true labels.

        Args:
            pred_labels (array-like): Predicted labels.
            Y_test (array-like): True labels.
            dm_i: Decision metric index.

        Returns:
            dict: A dictionary with 'correct' and 'incorrect' keys.
        """
        extracted = {'correct': [], 'incorrect': []}
        for i, (pred, true) in enumerate(zip(pred_labels, Y_test)):
            if pred == true:
                extracted['correct'].append(dm_i[i])
            else:
                extracted['incorrect'].append(dm_i[i])
        return extracted
    
    def _get_Us(self,estimator):
        """
        get the Us from the estimator.
        """
        Us_dict = {i:estimator.estimators_[i].Us_ for i in range(len(estimator.estimators_))}

        return Us_dict
    
    def _set_Us(self,estimator,Us_dict):
        """
        set the Us to the estimator.
        """
        for i in range(len(estimator.estimators_)):
            setattr(estimator.estimators_[i],'Us_',Us_dict[i])
        return None
    
    def _get_templates(self,estimator):
        templates_dict = {i:estimator.estimators_[i].templates_ for i in range(len(estimator.estimators_))}
        return templates_dict
    
    def _set_templates(self,estimator,templates_dict,duration):
        for i in range(len(estimator.estimators_)):
            setattr(estimator.estimators_[i],'templates_',templates_dict[i][:,:,:int(duration*self.srate)])
        return None
    
    def pre_train(self,X,Y,Yf=None):
        """
        Pre-trains the estimator using the provided data.

        Args:
            X (array-like): Training data.
            Y (array-like): Training labels.
            Yf (array-like, optional): Additional training data. Defaults to None.

        Returns:
            object: The pre-trained estimator.
        """
        estimator = clone(self.decoder).fit(X,Y,Yf=Yf)
        max_Us_dict = self._get_Us(estimator)
        max_templates_dict = self._get_templates(estimator)
        self.model_dict['pre_train'] = {'Us':max_Us_dict,'templates':max_templates_dict,'estimator':estimator}
        return estimator
    
    def train(self,X,Y,duration,Yf=None,filename=None):
        """
        Trains the KDE model and estimator using the provided data.

        Args:
            X (array-like): Training data.
            Y (array-like): Training labels.
            duration (float): Duration for which the model is trained.
            Yf (array-like, optional): Additional training data. Defaults to None.

        Returns:
            tuple: KDE models for correct and incorrect decisions, prior probability, and decision metrics.
        """
        if self.user_mode == 1 and filename is None:
            raise ValueError("Filename must be provided when user_mode is 1")
        data = X
        label = Y
        Yf = Yf
        spliter = EnhancedLeaveOneGroupOut(return_validate=False)
        aggregated_dm = {'correct': [], 'incorrect': []}  # 初始化空字典
        prob_list = []
        for train_ind, test_ind in spliter.split(data, y=label):
            X_train, Y_train = np.copy(data[train_ind]), np.copy(label[train_ind])
            X_test, Y_test = np.copy(data[test_ind]), np.copy(label[test_ind])
            model = clone(self.decoder).fit(X_train, Y_train, Yf=Yf) 
            pred_labels = model.predict(X_test)
            rhos = model.transform(X_test)
            rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
            
            dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] for i in rho_i])
            extracted_dm = self._extract_dm(pred_labels,Y_test,dm_i)
            sub_prob = len(extracted_dm['correct'])/(len(extracted_dm['correct'])+
                                              len(extracted_dm['incorrect']))
            prob_list.append(sub_prob)
            for key in aggregated_dm:
                aggregated_dm[key].extend(extracted_dm[key])
        dm0 = aggregated_dm['correct']     
        dm1 = aggregated_dm['incorrect']
        
        kde0 = gaussian_kde(dm0)
        if len(dm1) > 2:
            kde1 = gaussian_kde(dm1)
        else:
            kde1 = DummyKDE(0)
        prob = np.mean(prob_list)
        
        self.model_dict[duration] = {'kde0': kde0, 'kde1': kde1, 'prob': prob}
        if self.user_mode == 1 and filename is not None:
            self._save_model(filename)
        return kde0,kde1,prob
    
    def _get_model(self,duration):
        """
        Retrieves the model information for a given duration.

        Args:
            duration (float): Duration for which the model is trained.

        Returns:
            tuple: KDE models for correct and incorrect decisions, prior probability, and estimator.
        """
        model_info = self.model_dict[duration]
        kde0 = model_info['kde0']
        kde1 = model_info['kde1']
        prob = model_info['prob']
        Us_dict = self.model_dict['pre_train']['Us']
        templates_dict = self.model_dict['pre_train']['templates']
        estimator = self.model_dict['pre_train']['estimator']
        return kde0,kde1,prob,Us_dict,templates_dict,estimator
    
    def validate(self,testX,duration):
        """
        Validates the KDE model using the provided test data.

        Args:
            testX (array-like): Test data.
            duration (float): Duration for which the model is validated.

        Returns:
            float: Ratio of correctly classified samples.
        """
        if duration in self.model_dict:
            kde0,kde1,_,Us_dict,templates_dict,estimator = self._get_model(duration)
            self._set_Us(estimator,Us_dict)
            self._set_templates(estimator,templates_dict,duration)
            
            rhos = estimator.transform(testX)
            rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
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

    def decide(self,data,duration,P_thre = 0.95,filename=None):
        """
        Makes a decision based on the provided data and model.

        Args:
            data (array-like): Input data.
            duration (float): Duration for which the model is used.
            t_max (float, optional): Maximum duration. 
            p_thre (float, optional): Probability threshold. 
                                      It denpends on different Decoder.
                                      As to TRCA, Defaults to 0.95.

        Returns:
            tuple: Decision (True/False) and predicted label.
        """
        if self.user_mode == 1 and filename is None:
            raise ValueError("Filename must be provided when user_mode is 1")
        elif self.user_mode == 1 and filename is not None:
            self._load_model(filename)
        
        if duration in self.model_dict:
            kde0,kde1,prob,Us_dict,templates_dict,estimator = self._get_model(duration)
            
            self._set_Us(estimator,Us_dict)
            self._set_templates(estimator,templates_dict,duration)
            
            rhos = estimator.transform(data)
            label = estimator.predict(data)
            rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
            dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] for i in rho_i])
            p_H0 = kde0(dm_i)
            p_H1 = kde1(dm_i)
            p_pre = prob*p_H0/(prob*p_H0+(1-prob)*p_H1)
            p_thre = P_thre
            
            if p_pre >= p_thre or duration >= self.t_max :
                return True,label
            else:
                return False,label
        else:
            raise ValueError(f"No model found for duration: {duration}")