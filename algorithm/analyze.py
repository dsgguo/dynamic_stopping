import math

def acc(p_label:list, t_label:list):
    wrong = 0
    for label1,label2 in zip(p_label,t_label):
        if label1 != label2:
            wrong += 1
    acc = 1 - wrong/len(p_label)
    return acc

def ITR(M:int,P:float,T:float):
    def log2(x):
        return math.log(x,2)
    itr = (log2(M) + P*log2(P) + (1-P)*log2((1-P)/(M-1)) )*60/T
    return itr