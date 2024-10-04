import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import random
import tqdm
from torch import nn
class LearningModule:
    def __init__(self, data_processor, model, dataset,task_type, train_split, device, optimizer, criterion,log_func):
        self.data_processor = data_processor
        self.model = model
        self.dataset = dataset
        self.task_type = task_type
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.log_func = log_func

        self.date_list = self.dataset['date'].sort_values().unique()
        split_index = int(train_split * len(self.date_list))
        self.train_dates = self.date_list[:split_index]
        self.eval_dates = self.date_list[split_index:]

        self.train_losses = []
        self.eval_losses = []

    def process_batch(self, date):
        race_ids = self.dataset[self.dataset['date'] == date].index.unique()
        horse_ids, horse_data, horse_features, env_data, jockey_ids, jockey_data, labels, prev_race_data,tansho_batch,return_tansho_batch,return_hukusho_batch = self.data_processor.process_minibatch(race_ids)

        # Device transfer
        inputs = {
            'horse_data': horse_data.to(self.device),
            'jockey_data': jockey_data.to(self.device),
            'env_data': env_data.to(self.device),
            'horse_features': horse_features.to(self.device),
            'prev_race_data': prev_race_data.to(self.device),
        }

        predictions, updated_horse_data, updated_jockey_data,final_residual_output = self.model(
            inputs['horse_data'], inputs['jockey_data'], inputs['env_data'], 
            inputs['horse_features'], inputs['prev_race_data']
        )
        if self.task_type == 'regression':
            labels = labels.view(-1).to(self.device)
            predictions = predictions.view(-1).to(self.device)
            loss = self.criterion(predictions, labels)
        elif self.task_type == 'binary':
            labels = labels.view(-1).to(self.device)
            predictions = predictions.view(-1,2).to(self.device)
            loss = self.criterion(predictions, labels)
        elif self.task_type == 'list_net':
            labels = labels.to(self.device)
            predictions = predictions.view(-1,self.data_processor.num_selections).to(self.device)
            loss = self.criterion(predictions, labels)
            predictions = predictions.view(-1)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}") 
        
        self.data_processor.update_features(updated_horse_data.cpu(), horse_ids, self.data_processor.horse_features)
        self.data_processor.update_features(updated_jockey_data.cpu(), jockey_ids, self.data_processor.jockey_features)

        return horse_ids, predictions, labels, loss
    
    def merge_pred_results(self, horse_ids, pred, date):
        horse_ids = horse_ids.view(-1, 1)
        horse_ids = horse_ids.squeeze(1)
        horse_ids_np = horse_ids.cpu().numpy()
        pred_np = pred.cpu().detach().numpy()
        if self.task_type == 'binary':
            df = pd.DataFrame({
                'horse_id': horse_ids_np,
                'pred': pred_np[:,1],
            })
        else:
            df = pd.DataFrame({
                'horse_id': horse_ids_np,
                'pred': pred_np,
            })
        # horse_idごとの平均値を計算
        df['date'] = date#pd.to_datetime(date, format='%Y-%m-%d')
        df_results = df.groupby(['horse_id','date'], as_index=False)['pred'].mean()
        return df_results

    def process_pred_table(self,results):
        df = pd.merge(self.dataset.reset_index(), results, on=['horse_id', 'date'])
        return df

    def train(self, log_interval=100):
        self.data_processor.reset_features()
        self.model.train()
        mean_loss = 0

        for i, date in enumerate(tqdm.tqdm(self.train_dates)):
            horse_id, pred, label, loss = self.process_batch(date)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

            mean_loss += (loss.item() - mean_loss) / (i + 1)
            df = self.merge_pred_results(horse_id, pred, date)
            df = self.process_pred_table(df)
            if i % log_interval == log_interval-1:
                
                print(f'Loss: {mean_loss:.4f}')
                self.log_func(df,self.task_type)
                print(pred[:3])
                print('--------------')

        self.train_losses.append(mean_loss)
        return mean_loss

    def evaluate(self, log_interval=50):
        self.model.eval()
        mean_loss = 0
        df_list = pd.DataFrame([])

        with torch.no_grad():
            for i, date in enumerate(tqdm.tqdm(self.eval_dates)):
                horse_id, pred, label, loss = self.process_batch(date)
                mean_loss += (loss.item() - mean_loss) / (i + 1)
                df = self.merge_pred_results(horse_id, pred, date)
                df = self.process_pred_table(df)
                df_list = pd.concat([df_list,df], axis=0)
                if i % log_interval == log_interval-1:
                    print(f'Loss: {mean_loss:.4f}')
                    self.log_func(df,self.task_type)
                    print(pred[:3])
                    print('--------------')
                

        self.log_func(df_list,self.task_type)

        self.eval_losses.append(mean_loss)
        return mean_loss,df_list
    
    def test(self):
        self.model.eval()
        mean_loss = 0
        df_list = pd.DataFrame([])

        with torch.no_grad():
            for i, date in enumerate((self.date_list)):
                horse_id, pred, label, loss = self.process_batch(date)
                mean_loss += (loss.item() - mean_loss) / (i + 1)
                df = self.merge_pred_results(horse_id, pred, date)
                df = self.process_pred_table(df)
                df_list = pd.concat([df_list,df], axis=0)
                

        self.log_func(df_list,self.task_type)

        self.eval_losses.append(mean_loss)
        return mean_loss,df_list

