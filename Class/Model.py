import torch
import torch.nn.functional as F
import random
from torch import nn

class FeedForwardLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.9):
        super(ResidualBlock, self).__init__()
        self.feedforward = FeedForwardLayer(input_dim, output_dim)
        self.linear = nn.Linear(output_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.1)
        self.dropout_prob = dropout_prob
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        identity = x
        
        if self.training:
            if random.random() < self.dropout_prob:
                out = F.relu(self.feedforward(x))
                out = self.linear(out)
                out = self.batch_norm(out)
            else:
                out = self.shortcut(x) if self.shortcut is not None else x
        else:
            out = F.relu(self.feedforward(x))
            out = self.linear(out)
            out = self.batch_norm(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out += identity
        out = self.dropout(out)

        return out

class ResidualLayerBlock(nn.Module):
    def __init__(self, layer_dims):
        super(ResidualLayerBlock, self).__init__()
        self.input_dim = layer_dims[0]
        self.layers = self._build_layers(ResidualBlock, layer_dims)

    def _build_layers(self, block, layer_dims):
        layers = []
        num_layers = len(layer_dims)
        input_dim = self.input_dim
        for i, output_dim in enumerate(layer_dims):
            dropout_prob = 1 - i / (2 * num_layers)
            layers.append(block(input_dim, output_dim, dropout_prob))
            input_dim = output_dim
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class EncoderLayer(nn.Module):
    def __init__(self, layer_dims):
        super(EncoderLayer, self).__init__()
        self.input_dim = layer_dims[0]
        self.layers = self._build_layers(FeedForwardLayer, layer_dims)

    def _build_layers(self, block, layer_dims):
        layers = []
        input_dim = self.input_dim
        for output_dim in layer_dims:
            layers.append(block(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        del layers[-1]  # Remove last ReLU to maintain final structure
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class BaseHorseRacePredictor(nn.Module):
    def __init__(self, device, data_processor, hidden_size, env_input_dim, env_output_dim, horse_feature_input_dim, horse_feature_output_dim):
        super(BaseHorseRacePredictor, self).__init__()
        self.device = device
        self.num_horses = data_processor.num_horses
        self.num_selections = data_processor.num_selections
        self.hidden_size = hidden_size

        # エンコードレイヤー
        self.env_encoder = EncoderLayer([env_input_dim, 32, env_output_dim])
        self.horse_feature_encoder = EncoderLayer([horse_feature_input_dim, 32, horse_feature_output_dim])

        # 残差ブロック
        self.horse_residual_block = ResidualLayerBlock([hidden_size + env_output_dim + horse_feature_output_dim + 1, 128, 64, hidden_size])
        self.jockey_residual_block = ResidualLayerBlock([hidden_size * 2, 128, 64, hidden_size])
        self.race_residual_block = ResidualLayerBlock([hidden_size * self.num_selections, 128, 64, hidden_size * 3])
        self.listwise_residual_block = ResidualLayerBlock([hidden_size * 4, 128, 64, 32, 16, 8])

    def forward(self, horse_data, jockey_data, env_data, horse_features, prev_race_results):
        batch_size, seq_len, _ = horse_data.size()

        # 環境データと馬の特徴をエンコード
        env_data = env_data.view(-1, env_data.shape[-1])
        encoded_env = self.env_encoder(env_data)

        horse_features = horse_features.view(-1, horse_features.shape[-1])
        encoded_horse_features = self.horse_feature_encoder(horse_features)

        # 馬データをフラット化して組み合わせ
        horse_data_flat = horse_data.view(-1, self.hidden_size)
        prev_race_results = prev_race_results.view(-1, 1)
        combined_horse_data = torch.cat([horse_data_flat, prev_race_results, encoded_horse_features, encoded_env], dim=1)

        # 馬、騎手、レースデータを順次処理
        horse_residual_output = self.horse_residual_block(combined_horse_data)
        jockey_data_flat = jockey_data.view(-1, self.hidden_size)
        combined_jockey_data = torch.cat([horse_residual_output, jockey_data_flat], dim=1)
        jockey_residual_output = self.jockey_residual_block(combined_jockey_data)

        race_data = jockey_residual_output.view(batch_size, -1)
        race_residual_output = self.race_residual_block(race_data)
        race_residual_output = race_residual_output.unsqueeze(1).repeat(1, self.num_selections, 1).view(-1, race_residual_output.shape[-1])

        final_data = torch.cat([race_residual_output, jockey_residual_output], dim=1)
        final_residual_output = self.listwise_residual_block(final_data)
        
        updated_horse_data = horse_residual_output.detach().clone().reshape(horse_data.size())
        updated_jockey_data = jockey_residual_output.detach().clone().reshape(jockey_data.size())


        # 中間結果を返す（出力クラスで利用する）
        return final_residual_output, updated_horse_data, updated_jockey_data


class RegressionOutput(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(RegressionOutput, self).__init__()
        self.fc_output = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc_output(x)
    
class ClassificationOutput(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationOutput, self).__init__()
        self.fc_output = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc_output(x)
    
class HorseRaceModel(nn.Module):
    def __init__(self, task_type, device, data_processor, hidden_size, env_input_dim, env_output_dim, horse_feature_input_dim, horse_feature_output_dim, output_dim=1, num_classes=2):
        super(HorseRaceModel, self).__init__()
        
        # ベースのモデル（共通の処理部分）
        self.base_model = BaseHorseRacePredictor(device, data_processor, hidden_size, env_input_dim, env_output_dim, horse_feature_input_dim, horse_feature_output_dim)
        
        # タスクに応じた出力クラスの選択
        if task_type == 'regression':
            self.output_layer = RegressionOutput(8, 1)
        elif task_type == 'binary':
            self.output_layer = ClassificationOutput(8, 2)
        elif task_type == 'list_net':
            self.output_layer = RegressionOutput(8, 1)       
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def forward(self, horse_data, jockey_data, env_data, horse_features, prev_race_results):
        # ベースの処理を実行
        final_residual_output, updated_horse_data, updated_jockey_data = self.base_model(horse_data, jockey_data, env_data, horse_features, prev_race_results)
        
        # タスクに応じた出力を取得
        predictions = self.output_layer(final_residual_output)
        return predictions, updated_horse_data, updated_jockey_data,final_residual_output