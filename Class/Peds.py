import pandas as pd
import tqdm
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from urllib.request import urlopen

class Peds:
    def __init__(self,peds):
        self.peds = peds
        self.peds_e = pd.DataFrame()
        self.le_peds = None
    
    @classmethod
    def read_pickle(cls,path_list):
        df = pd.concat([pd.read_pickle(path) for path in path_list]) 
        return cls(df)
    
    @staticmethod
    def scrape(horse_id_list,pre_df = None,save_path = 'pickle/peds_temp' ):
        peds = {}
        if pre_df is not None:
            horse_id_list = [horse_id  for horse_id in horse_id_list if not horse_id in pre_df.index.unique()]
        for horse_id in tqdm.tqdm(horse_id_list):
            if horse_id in peds.keys():
                continue
            try:
                url = 'https://db.netkeiba.com/horse/ped/'+horse_id
                html = urlopen(url).read()
                df = pd.read_html(html)[0]
                generations = {}
                for i in reversed(range(5)):
                    generations[i] = df[i]
                    df.drop([i],axis=1,inplace=True)
                    df = df.drop_duplicates()
                ped = pd.concat([generations[i] for i in range(5)]).rename(horse_id)
                peds[horse_id] = ped.reset_index(drop=True)
                time.sleep(1)
            except IndexError:
                continue
            except: 
                break

        if not len(peds) == 0:
            peds = pd.concat([peds[horse_id] for horse_id in peds],axis=1).T
            peds = peds.add_prefix('peds_')

            peds = pd.concat([pre_df,peds],axis=0)

            peds.to_pickle(save_path)
        else:
            peds = pre_df
        return peds
    
    def encode(self):
        df = self.peds.copy()
        all_data = df.values.ravel()
        if self.le_peds is None:
            self.le_peds = LabelEncoder().fit(all_data)
            encoded_data = self.le_peds.transform(all_data)
            self.peds_e = pd.DataFrame(encoded_data.reshape(df.shape),index=df.index, columns=df.columns)
        else:
            all_data = pd.Series(all_data)
            mask_peds = all_data.isin(self.le_peds.classes_)
            new_peds_id = all_data.mask(mask_peds).dropna().unique()
            self.le_peds.classes_ = np.concatenate([self.le_peds.classes_,new_peds_id])
            encoded_data = self.le_peds.transform(all_data)
            self.peds_e = pd.DataFrame(encoded_data.reshape(df.shape),index=df.index, columns=df.columns)
            print('既存のエンコーディングを引き継ぎました')
            

    def get_no_peds_list(self,data_p,save_path ='pickle/peds_temp'):
        self.no_peds_list = data_p[~data_p['horse_id'].isin(self.peds.index.unique())]['horse_id'].unique()
        self.peds = self.scrape(self.no_peds_list,self.peds,save_path)
