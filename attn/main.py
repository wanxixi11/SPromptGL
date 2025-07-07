import numpy as np
import torch
import scipy.io as io

# 读取.npy文件
data = np.load('/DATA/wanxixi/baselineabide1/attn/attn_map_ABIDE.npy')
print(data.shape)
data = np.array(data)
        
mat_path = '/DATA/wanxixi/baselineabide1/attn//map.mat'
io.savemat(mat_path, {'map': data})
