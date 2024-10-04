import numpy as np
import torch
import torch.nn.functional as F
import random
from torch import nn


def get_ped_one_hot(data,p):
    peds_columns=data.columns[data.columns.str.contains('peds_')]
    peds_num = p.peds_e.values.max()+1
    one_hot_size = peds_num+1
    horse_id_list = data['horse_id'].unique()
    horse_num = len(horse_id_list)
    horse_max_id = data['horse_id'].max()

    peds_one_hot = torch.zeros((horse_max_id+1,one_hot_size))
    for horse_id in horse_id_list:
        peds_data = data[data['horse_id']==horse_id].iloc[0][peds_columns].to_numpy(dtype='float32')
        peds_data=np.nan_to_num(peds_data,nan=peds_num)
        peds_data = torch.from_numpy(peds_data).to(torch.int64)


        one_hot=F.one_hot(peds_data,num_classes=one_hot_size)

        new_one_hot = torch.zeros(one_hot.shape[1],dtype=torch.float32)
        for i in range(len(one_hot)):
            generation = int(i/2)+1
            new_one_hot += one_hot[i]/generation
        peds_one_hot[horse_id] = new_one_hot
    return peds_one_hot




def leftShiftIndex(arr, n):
   result = np.concatenate([arr[n:], arr[:n]])
   return result

def expand_id_list(arr_len,select_num = 14):
    arr = np.arange(arr_len)
    expanded_id_list = []
    random.shuffle(arr)
    for i in range(arr_len):
        shiht_id_list = leftShiftIndex(arr,i)
        expanded_id_list.append(shiht_id_list)
    expanded_id_list = np.array(expanded_id_list)
    if arr_len < select_num:
        add_per = int(select_num/arr_len)+1
        selected_id_list = np.tile(expanded_id_list, (1, add_per))[:,:select_num]
    else:
        selected_id_list = expanded_id_list[:,:select_num]
    return selected_id_list


from Func.ListWise_func import listnet_loss

def set_criterion(task_type):
    if task_type == 'regression':
        criterion = nn.MSELoss()
    elif task_type == 'binary':
        criterion = nn.CrossEntropyLoss()
    elif task_type == 'list_net':
        criterion = listnet_loss
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return criterion

def log_func(df,task_type):
    if task_type == 'regression':
        df['rank'] = df.groupby('index')['pred'].rank(ascending=True)
    else:
        df['rank'] = df.groupby('index')['pred'].rank(ascending=False)

    df_rank1 = df[df['rank'] == 1]
    df_rank1_high_return = df_rank1[df_rank1['単勝']>8]

    race_num = len(df['index'].unique())
    bet_num = len(df_rank1_high_return)

    correct_data = df_rank1[df_rank1['着順'] == 1]
    hukusho_correct_data = df_rank1[df_rank1['着順'] <=3]
    correct_high_return = df_rank1_high_return[df_rank1_high_return['着順'] == 1]
    correct_high_return_hukusho = df_rank1_high_return[df_rank1_high_return['着順'] <= 3]

    accuracy = len(correct_data) / len(df_rank1) * 100 if len(df_rank1) > 0 else 0
    accuracy_hukusho = len(hukusho_correct_data) / len(df_rank1) * 100 if len(df_rank1) > 0 else 0
    
    return_money_tansho = correct_high_return['tansho_return'].sum()/len(df_rank1_high_return) * 100 if bet_num > 0 else 100
    return_money_hukusho = correct_high_return_hukusho['hukusho_return'].sum()/bet_num * 100 if bet_num > 0 else 100


    print(f'race_num {race_num} | bet_num: {bet_num} | per: {bet_num/race_num*100:.1f}%')
    print(f'Tansho:   Accuracy: {accuracy:.1f}% | Return_money: {return_money_tansho:.1f}%')
    print(f'Hukusho:  Accuracy: {accuracy_hukusho:.1f}% | Return_money: {return_money_hukusho:.1f}%')
    
    return accuracy,return_money_tansho

import os

def create_directory(base_path,folder_name):
    """
    指定されたフォルダが存在しない場合、新しく作成する関数。
    
    Parameters:
    - base_path: フォルダのベースとなるパス
    - folder_name_template: フォルダ名のテンプレート（例: "folder_{}"）
    - args: テンプレートに埋め込む変数
    
    Returns:
    - 完全なフォルダパスを返す
    """
    # 完全なフォルダパス
    full_path = os.path.join(base_path, folder_name)
    
    # フォルダが存在しない場合は作成
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"フォルダ '{full_path}' を作成しました。")
    else:
        print(f"フォルダ '{full_path}' は既に存在します。")
    
    return full_path

import os
import torch.optim as optim
def create_predictionTable(base_path,folder_name,data,RacingDataProcessor,LearningModule):
    file_list = os.listdir(base_path+'/'+folder_name)
    file_name = file_list[-1]
    parameters = [item.split('_',1)[1] for item in folder_name.split('__')]
    print(parameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # デバイスの確認

    task_type = parameters[0]
    h_size = int(parameters[1])
    num_selections=int(parameters[2])
    pre_race_rate = float(parameters[3])

    rdp = RacingDataProcessor(data,h_size,num_selections,pre_race_rate,task_type)

    # net = HorseRaceModel(task_type, device,rdp, h_size,E_in,E_out, dh_in,dh_out).to(device)
    net = torch.load('{}/{}/{}'.format(base_path,folder_name,file_name))
    # 損失関数の定義
    criterion = set_criterion(task_type)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_split = 0.7
    lem = LearningModule(rdp,net,data,task_type,train_split,device,optimizer, criterion,log_func)

    mean_loss,df = lem.test()
    return df