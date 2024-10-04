
import pandas as pd
import tqdm
import numpy as np
import time
from  urllib.request import urlopen

class Return:
    def __init__(self,return_tables):
        self.return_tables = return_tables
        # self.hukusho = self.return_tables[self.return_tables[0]=='複勝'][[1,2]]
    
    @classmethod
    def read_pickle(cls,path_list):
        df = pd.concat([pd.read_pickle(path) for path in path_list]) 
        return cls(df)
    
    @staticmethod
    def scrape(race_id_list,pre_df):
        return_tables = {}
        if pre_df is not None:
            pre_list=pre_df.index.unique()
            race_id_list = [race_id  for race_id in race_id_list if not race_id in pre_list]
        for race_id in tqdm.tqdm(race_id_list):
            try:
                url = 'https://db.netkeiba.com/race/'+race_id
                f = urlopen(url)
                html = f.read()
                html = html.replace(b'<br />', b'br')
                dfs = pd.read_html(url)
                return_tables[race_id]  = pd.concat([dfs[1],dfs[2]])
                time.sleep(1)
            except IndexError:
                continue
            except Exception as e:
                print(e)
            break
        
        for key in return_tables.keys():
            return_tables[key].index = [key]*len(return_tables[key])

        return_tables = pd.concat([return_tables[key] for key in return_tables.keys()],sort=False)
        df_return_tables = pd.DataFrame(return_tables)

        df_return_tables = pd.concat([pre_df,df_return_tables],axis=0)
        

        return df_return_tables
            
    @property
    def hukusho(self):
        hukusho = self.return_tables[self.return_tables[0]=='複勝'][[1,2]]
        wins = hukusho[1].str.split(' ',expand=True).drop([2,3],axis=1)
        wins.columns = ['win_0','win_1','win_2']
        returns = hukusho[2].str.split(' ',expand=True).drop([2,3],axis=1)
        returns.columns = ['return_0','return_1','return_2']
        df = pd.concat([wins,returns],axis=1)
        for column in df.columns:
            df[column] = df[column].str.replace(',','')
        return df.fillna(0).astype(int) 

    @property
    def tansho(self):
        tansho = self.return_tables[self.return_tables[0]=='単勝'][[1,2]]
        tansho.columns = ['win','return']

        for column in tansho.columns:
            tansho[column] = pd.to_numeric(tansho[column],errors='coerce')

        return tansho
