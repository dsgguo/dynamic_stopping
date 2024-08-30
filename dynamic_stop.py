# -*- coding: utf-8 -*-
#
# Authors: Duan Shunguo<dsg@tju.edu.cn>
# Date: 2024/9/1
# License: MIT License

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

data = np.random.normal(0, 1, 1000)

kde = gaussian_kde(data)
