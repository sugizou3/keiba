import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class ScrapedDataProcessor:
    def __init__(self):
        self.data = pd.DataFrame()
        self.data_p = pd.DataFrame()
        self.data_pe = pd.DataFrame()
        self.data_c = pd.DataFrame()
        self.le_horse = None
        self.le_jockey = None
        self.le_R = None
        self.le_horse = None
        self.le_venue = None
    
    def read_encode_data(self,le):
        self.le_horse = le.le_horse
        self.le_jockey = le.le_jockey
        self.le_R = le.le_R
        self.le_horse = le.le_horse
        self.le_venue = le.le_venue

    def get_previous_result(self):
        self.data_p = self.data_p.reset_index()  # インデックスにレースIDが入っていると仮定

        df=self.data_p[['horse_id','date','着順']].sort_values(by=['horse_id','date'])
        df['前走の着順'] = df.groupby('horse_id')['着順'].shift(1)

        # dfの前走の着順をhorse_idとdateをキーにしてマージ
        self.data_p = self.data_p.merge(df[['horse_id', 'date', '前走の着順']], on=['horse_id', 'date'], how='left')

        # NaNになった箇所を10で埋める
        self.data_p['前走の着順'] = self.data_p['前走の着順'].fillna(10)

                # レースIDを再度インデックスに設定
        self.data_p = self.data_p.set_index('index')

        return self.data_p


        
    
    def merge_peds(self,peds):
        self.data_pe = self.data_p.merge(peds,left_on='horse_id',right_index=True,how='left')
    
    def to_normalization(self,column):
        df = self.data_pe[column]
        df = (df - df.mean()) / df.std()
        return df
    
    def process_categorical(self):
        if self.le_horse is None:
            self.le_horse = LabelEncoder().fit(self.data_pe['horse_id'])
        if self.le_jockey is None:
            self.le_jockey = LabelEncoder().fit(self.data_pe['jockey_id'])
        if self.le_R is None:
            self.le_R = LabelEncoder().fit(self.data_pe['R'])
        if self.le_venue is None:
            self.le_venue = LabelEncoder().fit(self.data_pe['開催地'])
        df = self.data_pe.copy()

        mask_horse = df['horse_id'].isin(self.le_horse.classes_)
        new_horse_id = df['horse_id'].mask(mask_horse).dropna().unique()
        self.le_horse.classes_ = np.concatenate([self.le_horse.classes_,new_horse_id])
        df['horse_id'] = self.le_horse.transform(df['horse_id'])
        
        mask_jockey = df['jockey_id'].isin(self.le_jockey.classes_)
        new_jockey_id = df['jockey_id'].mask(mask_jockey).dropna().unique()
        self.le_jockey.classes_ = np.concatenate([self.le_jockey.classes_,new_jockey_id])
        df['jockey_id'] = self.le_jockey.transform(df['jockey_id'])

        weather = self.data_pe['weather'].unique()
        race_type = self.data_pe['race_type'].unique()
        ground_state = self.data_pe['ground_state'].unique()
        sexes = self.data_pe['性'].unique()
        venue = self.data_pe['開催地'].unique()
        R_num = self.data_pe['R'].unique()

        df['weather'] = pd.Categorical(df['weather'],weather)
        df['race_type'] = pd.Categorical(df['race_type'],race_type)
        df['ground_state'] = pd.Categorical(df['ground_state'],ground_state)
        df['性'] = pd.Categorical(df['性'],sexes)
        df['開催地'] = pd.Categorical(df['開催地'],venue)
        df['R'] = pd.Categorical(df['R'],R_num)

        df =pd.get_dummies(df,columns=['weather','race_type','ground_state','性','R','開催地'],dtype='uint8')
        
        df['枠番'] = self.to_normalization('枠番')
        df['斤量'] = self.to_normalization('斤量')
        df['タイム'] = self.to_normalization('タイム')
        df['course_len'] = self.to_normalization('course_len')
        df['年齢'] = self.to_normalization('年齢')
        df['体重'] = self.to_normalization('体重')
        df['horse_len'] = self.to_normalization('horse_len')
        df['体重変化'] = self.to_normalization('体重変化')
        df['単勝_norm'] = self.to_normalization('単勝')
        df['corse_used_num'] = self.to_normalization('corse_used_num')
        
        
        
        self.data_c = df
    
    def get_return_info(self,rt):
        self.data_r = pd.merge(
            self.data_c.reset_index(),
            rt.tansho.reset_index().rename(columns={'win': '馬番'})[['index', 'return', '馬番']],
            on=['index', '馬番'],
            how='left'
        ).fillna(0)

        # 複勝データ (rt.hukusho) を一度に縦持ちに変換
        hukusho_merged = pd.wide_to_long(
            rt.hukusho.reset_index(),
            stubnames=['win', 'return'],
            i='index',
            j='num',
            sep='_'
        ).reset_index().rename(columns={'win': '馬番', 'return': 'hukusho_return'})

        # 単勝データと複勝データをマージし、複勝リターンを結合
        self.data_r = pd.merge(self.data_r, hukusho_merged[['index', '馬番', 'hukusho_return']], on=['index', '馬番'], how='left')

        self.data_r = self.data_r.fillna(0).set_index('index')
        self.data_r = self.data_r.rename(columns={'return': 'tansho_return'})
        self.data_r['tansho_return'] = self.data_r['tansho_return']*0.01
        self.data_r['hukusho_return'] = self.data_r['hukusho_return']*0.01
    