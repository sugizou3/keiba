import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import random

from torch import nn

# def leftShiftIndex(arr, n):
#    result = np.concatenate([arr[n:], arr[:n]])
#    return result

# def expand_id_list(arr_len,select_num = 14):
#     arr = np.arange(arr_len)
#     expanded_id_list = []
#     random.shuffle(arr)
#     for i in range(arr_len):
#         shiht_id_list = leftShiftIndex(arr,i)
#         expanded_id_list.append(shiht_id_list)
#     expanded_id_list = np.array(expanded_id_list)
#     if arr_len < select_num:
#         add_per = int(select_num/arr_len)+1
#         selected_id_list = np.tile(expanded_id_list, (1, add_per))[:,:select_num]
#     else:
#         selected_id_list = expanded_id_list[:,:select_num]
#     return selected_id_list

def listnet_loss(pred, y):
    # オーバーフローを防ぐためにlogsumexpを使用
    pred_log_softmax = pred - torch.logsumexp(pred, dim=1, keepdim=True)
    y_softmax = F.softmax(y, dim=0)
    
    # クロスエントロピー損失を計算
    loss = -torch.sum(y_softmax * pred_log_softmax, dim=1)
    return loss.mean()




def ranking(data):
        # 大きい順に並べたときのインデックスを取得
    _, indices = torch.sort(data, descending=True)

    # インデックスに基づいて順位を計算
    ranks = torch.zeros_like(indices, dtype=torch.long)
    ranks[indices] = torch.arange(1, len(data)+1)
    return ranks









