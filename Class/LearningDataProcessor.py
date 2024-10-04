import numpy as np
import torch



class RacingDataProcessor:
    def __init__(self, data, feature_size, num_selections,pre_race_rate=0.5,task_type='regression'):
        self.data = data
        self.num_selections = num_selections
        self.task_type = task_type
        self.pre_race_rate = pre_race_rate

        # ID数の取得
        self.num_jockeys = data['jockey_id'].max() + 1
        self.num_horses = data['horse_id'].max() + 1

        # 初期化
        self.feature_size = feature_size
        self.jockey_features = torch.zeros(self.num_jockeys, feature_size, requires_grad=False)
        self.horse_features = torch.zeros(self.num_horses, feature_size, requires_grad=False)

        # カラムのセットアップ
        self.environment_columns = data.columns[data.columns.str.contains('weather_|ground_state_|開催地_|R_')].tolist() + \
                                   ['course_len',  'corse_used_num','horse_len']
        self.horse_columns = data.columns[data.columns.str.contains('性_')].tolist() + \
                             ['枠番','斤量', '年齢', '体重', '体重変化']

    def reset_features(self):
        """騎手と馬の特徴量をリセット"""
        self.jockey_features.zero_()
        self.horse_features.zero_()

    def generate_selection_matrix(self, total_length, num_selections=14):
        """選択リストの生成"""
        indices = np.arange(total_length)
        np.random.shuffle(indices)

        # total_length が num_selections より小さい場合、リピートして num_selections に合わせる
        if total_length < num_selections:
            repeat_factor = (num_selections // total_length) + 1  # 繰り返しの回数を計算
            extended_indices = np.tile(indices, repeat_factor)[:num_selections]
            selection_matrix = np.vstack([np.roll(extended_indices, i) for i in range(total_length)])
        else:
            selection_matrix = np.concatenate([np.roll(indices, i)[:num_selections] for i in range(total_length)], axis=0).reshape(total_length, num_selections)
        
        return selection_matrix

    def convert_to_tensor(self, df, columns):
        """Pandas DataFrameをTensorに変換"""
        return torch.tensor(df[columns].to_numpy(dtype='float32'))
    
    def get_tansho_info(self, df):
        """y__binary_in3をTensorに変換"""
        return torch.tensor(df['単勝_norm'].to_numpy(dtype='float32'))
    
    def get_tansho_return(self, df):
        """y__binary_in3をTensorに変換"""
        return torch.tensor(df['tansho_return'].to_numpy(dtype='float32'))
    
    def get_hukusho_return(self, df):
        """y__binary_in3をTensorに変換"""
        return torch.tensor(df['hukusho_return'].to_numpy(dtype='float32'))
    
    def get_target(self,df):
        if self.task_type == 'regression':
            return torch.tensor(df['y__reg_着順'].to_numpy(dtype='float32'))
        elif self.task_type == 'binary':
            return torch.tensor(df['y__binary_in3'].to_numpy(dtype='float32')).to(dtype=torch.long)
        elif self.task_type == 'list_net':
            return torch.tensor(2-df['着順'].to_numpy(dtype='float32')*0.1)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
            
        

    def get_previous_race_result(self, df):
        """前走の着順を処理してTensorに変換"""
        previous_result = 2 - df['前走の着順'].to_numpy(dtype='float32') * 0.1
        return torch.tensor(previous_result)

    def prepare_batch_data(self, horse_ids, jockey_ids, df):
        """共通するバッチデータの準備"""
        selection_matrix = self.generate_selection_matrix(len(horse_ids), self.num_selections)
        
        env_tensor = self.convert_to_tensor(df.iloc[0], self.environment_columns)
        env_batch = env_tensor.unsqueeze(0).expand(selection_matrix.shape[0], self.num_selections, -1)

        horse_id_batch = torch.tensor(horse_ids[selection_matrix])
        horse_feature_batch = self.horse_features[horse_ids][selection_matrix]
        jockey_id_batch = torch.tensor(jockey_ids[selection_matrix])
        jockey_feature_batch = self.jockey_features[jockey_ids][selection_matrix]

        return selection_matrix, env_batch, horse_feature_batch, horse_id_batch, jockey_feature_batch, jockey_id_batch

    def process_single_race(self, index):
        """単一レースのデータを取得"""
        df = self.data.loc[index]
        horse_ids = df['horse_id'].unique()
        jockey_ids = df['jockey_id'].unique()

        # バッチデータの準備
        selection_matrix, env_batch, horse_feature_batch, horse_id_batch, jockey_feature_batch, jockey_id_batch = \
            self.prepare_batch_data(horse_ids, jockey_ids, df)

        # 馬の特徴量とターゲットデータの計算
        horse_feature_tensor = self.convert_to_tensor(df, self.horse_columns).view(len(horse_ids), -1)
        target = self.get_target(df)
        tansho = self.get_tansho_info(df)
        return_tansho = self.get_tansho_return(df)
        return_hukusho = self.get_hukusho_return(df)
        previous_results = self.get_previous_race_result(df)

        # 選択されたリストに基づくバッチの作成
        horse_feature_batch_selected = horse_feature_tensor[selection_matrix]
        target_batch = target[selection_matrix]
        tansho_batch = tansho[selection_matrix]
        return_tansho_batch = return_tansho[selection_matrix]
        return_hukusho_batch = return_hukusho[selection_matrix]
        prev_result_batch = previous_results[selection_matrix]

        return horse_id_batch, horse_feature_batch, horse_feature_batch_selected, env_batch, jockey_id_batch, jockey_feature_batch, target_batch, prev_result_batch,tansho_batch,return_tansho_batch,return_hukusho_batch

    def process_minibatch(self, index_list):
        """ミニバッチでレースデータを処理"""
        results = [self.process_single_race(index) for index in index_list]

        # 結果の統合
        return map(lambda t: torch.cat(t, dim=0), zip(*results))

    def update_features(self, new_data, id_list, feature_list):
        """新しいデータに基づいて特徴量を更新"""
        unique_ids = torch.unique(id_list)
        id_data_dict = {id.item(): [] for id in unique_ids}

        for i, ids in enumerate(id_list):
            for j, id in enumerate(ids):
                id_data_dict[id.item()].append(new_data[i, j])

        # 特徴量の更新
        for id, data in id_data_dict.items():
            feature_list[id] = (self.pre_race_rate * feature_list[id] + torch.mean(torch.stack(data), dim=0))/(1+self.pre_race_rate)
