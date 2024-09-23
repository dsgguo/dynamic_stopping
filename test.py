import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

class DynamicStop:
    def __init__(self,decoder,train_data,train_label,duration):
        self.decoder = decoder#FBTRCA
        self.data = train_data
        self.label = train_label
        self.duration = duration
        

    def train(self):
        rhos = self.data
        rho_i = {i: rhos[i, :] for i , _ in enumerate(rhos)} 
        print(rho_i)
        dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] / sum(rho_i[i]) for i in rho_i])
        return dm_i
    
    def predict(self):
        pass
    
def extract_dm(pred_labels, Y_test, dm_i):
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
    
# np.random.seed(3)
# testX = np.random.rand(40, 40)

# # 创建 DynamicStop 类的实例
# dynamic_stop_instance = DynamicStop(None, testX, None,duration=1.0)

# # 调用 train 方法
# dm_i = dynamic_stop_instance.train()
# print("dm_i:", dm_i)

# # 对 dm_i 进行 KDE
# kde = gaussian_kde(dm_i)

# # 生成用于绘图的 x 轴数据
# x = np.linspace(min(dm_i), max(dm_i), 100)
# kde_values = kde(x)

# # 绘制 KDE 曲线
# plt.plot(x, kde_values)
# plt.xlabel('dm_i values')
# plt.ylabel('Density')
# plt.title('KDE of dm_i')
# plt.show()


