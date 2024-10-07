import numpy as np

# 示例数组


rhos = np.random.rand(40, 40)
rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 


dm0 = np.array([[np.partition(rho_i[i], -1)[-1], np.partition(rho_i[i], -2)[-2]] 
                 for i in rho_i])
dm1 = np.array([[np.partition(rho_i[i], -1)[-1], np.partition(rho_i[i], -2)[-2]] 
                 for i in rho_i])

L = np.concatenate((dm0,dm1),axis=0)
t_labels = np.concatenate((np.ones(len(dm0)), np.zeros(len(dm1))), axis=0)
print(t_labels)