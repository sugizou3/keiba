import pandas as pd
import numpy as np


class RaceFeature():
    def __init__(self,results):
        self.data = results[['着順', '単勝', '人気', 'date', 'course_len', 'weather', 'race_type',
                                'ground_state',  '年齢', 'R', 'corse_used_num','開催地']].copy()
        print(self.data['date'])
        self.data['出走数'] = self.data.groupby(level=0)['着順'].max()
        self.data = self.get_y()

        self.data_u = self.race_unique(self.data)
        self.data_c = pd.DataFrame()
        self.data_r = pd.DataFrame()
    
    def get_y(self):
        data = self.data.copy()
        df = self.data.groupby(level=0)[['着順','人気']].corr(method='spearman')
        df = df.reset_index()
        df.index = df['level_0']
        df = df['着順']
        data['corr'] = df[df.index.duplicated()]

        df = data[data['着順'] < 4]
        df2 = df['人気']/df['出走数']
        data['top3_popularity'] = df2.groupby(level=0).mean()

        df3 = df['人気']/df['出走数']/df['着順']
        data['top3_popularity_wmean'] = df3.groupby(level=0).sum()

        df3 = df['単勝']/df['着順']
        data['top3_tansho'] = df3.groupby(level=0).sum()
        return data
    
    def race_unique(self,data_in):
        data = data_in.copy()
        data['年齢_平均']=data.groupby(level=0)['年齢'].mean()  
        data['年齢_標準偏差']=data.groupby(level=0)['年齢'].std()
        data = data.drop(['着順', '単勝', '人気','年齢'],axis =1).drop_duplicates()  
        return data
    
    def process_categorical(self,data):   
        df = data.copy()

        weather = data['weather'].unique()
        race_type = data['race_type'].unique()
        ground_state = data['ground_state'].unique()

        venue = data['開催地'].unique()
        R_num = data['R'].unique()

        df['weather'] = pd.Categorical(df['weather'],weather)
        df['race_type'] = pd.Categorical(df['race_type'],race_type)
        df['ground_state'] = pd.Categorical(df['ground_state'],ground_state)
        df['開催地'] = pd.Categorical(df['開催地'],venue)
        df['R'] = pd.Categorical(df['R'],R_num)


        df =pd.get_dummies(df,columns=['weather','race_type','ground_state','開催地','R'],dtype='uint8')

        self.data_c = df


    





