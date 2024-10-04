import pandas as pd
import tqdm
import re
import numpy as np
import time
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from Class.ScrapedDataProcessor import ScrapedDataProcessor
from urllib.request import urlopen

venue = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}


class Results(ScrapedDataProcessor):
    def __init__(self,results):
        super(Results,self).__init__()
        self.data = results

    @classmethod
    def read_pickle(cls,path_list):
        df = pd.concat([pd.read_pickle(path) for path in path_list]) 
        return cls(df)
    
    @staticmethod    
    def scrape_preprocessing(results):
        df = results.copy()
        df = df.rename(columns={'着 順':'着順','枠 番':'枠番','馬 番':'馬番','人 気':'人気'})
        #着順の数字以外の文字列が含まれているものを取り除く
        # df = df[~(df["着順"].astype(str).str.contains("\D"))]
        
        df['着順'] = pd.to_numeric(df['着順'],errors='coerce')
        df.dropna(subset=['着順'],inplace=True)
        df["着順"] = df["着順"].astype(int)

        #性齢を性別と年齢に分ける
        df["性"] = df["性齢"].map(lambda x :str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x :str(x)[1:]).astype(int)

        #馬体重を体重と体重変化に分ける
        df['体重'] = df['馬体重'].str.split('(',expand=True)[0].astype(int)
        df['体重変化'] = df['馬体重'].str.split('(',expand=True)[1].str[:-1].astype(int)

        df = df.drop(['性齢','馬体重'],axis=1)

        #データをint,floatに変換
        df["単勝"] = df["単勝"].astype(float)

        #不要な列を削除
        # df.drop(['タイム','調教師','着差','馬体重','性齢'],axis=1,inplace=True)

        df['date'] = pd.to_datetime(df['date'],format='%Y年%m月%d日')

        return df

    @staticmethod
    def scrape(race_id_list,pre_df = None):
        race_results = {}
        race_infos = {}
        
        if pre_df is not None:
            race_id_list = pd.Series(race_id_list)
            maskdata = race_id_list.isin(pre_df.index.unique())
            race_id_list = race_id_list.mask(maskdata).dropna().unique()

        for race_id in tqdm.tqdm(race_id_list):
            try:
                url = 'https://db.netkeiba.com/race/'+race_id
                html = urlopen(url).read()
                df = pd.read_html(html)[0]

                #horse_idとJockeyIdをスクレイピング
                html = requests.get(url)
                html.encoding = 'EUC-JP'
                soup = BeautifulSoup(html.text,'html.parser')
                #horse_id
                horse_id_list = []
                horse_a_list = soup.find('table',attrs={'summary':'レース結果'}).find_all('a',attrs={'href':re.compile('^/horse')})
                for a in horse_a_list:
                    horse_id = re.findall(r'\d+',a['href'])
                    horse_id_list.append(horse_id[0])
                #jockey_id
                jockey_id_list = []
                jockey_a_list = soup.find('table',attrs={'summary':'レース結果'}).find_all('a',attrs={'href':re.compile('^/jockey')})
                for a in jockey_a_list:
                    jockey_id = re.findall(r'\d+',a['href'])
                    jockey_id_list.append(jockey_id[0])

                df['horse_id'] = horse_id_list
                df['jockey_id'] = jockey_id_list
                race_results[race_id] = df




                texts = soup.find('div',attrs ={'class':'data_intro'}).find_all('p')[0].text + \
                    soup.find('div',attrs ={'class':'data_intro'}).find_all('p')[1].text
                info = re.findall(r'\w+',texts)
                info_dict = {}
                for text in info:
                    if text in ['芝','ダート']:
                        info_dict['race_type'] = text
                    if '障' in text:
                        info_dict['race_type'] = '障害'
                    if 'm' in text:
                        info_dict['course_len'] = re.findall(r'\d+',text)[0]
                    if text in ['良','稍重','重','不良']:
                        info_dict['ground_state'] = text
                    if text in ['曇','晴','雨','小雨','小雪','雪']:
                        info_dict['weather'] = text
                    if '年' in text:
                        info_dict['date'] = text


                race_infos[race_id] = info_dict



                time.sleep(1)
            except IndexError:
                continue

            except:
                break


        if not len(race_results) == 0:
            for key in race_results.keys():
                race_results[key].index = [key]*len(race_results[key])

            race_results = pd.concat([race_results[key] for key in race_results.keys()],sort=False)
            results = pd.DataFrame(race_results)
            race_infos = pd.DataFrame(race_infos).T

            results_addinfo = results.merge(race_infos,left_index=True,right_index=True,how='inner')

            results_addinfo['R'] = results_addinfo.index.map(lambda x :str(x)[-2:])
            results_addinfo['corse_used_num']= results_addinfo.index.map(lambda x :int(str(x)[6:8]))
            results_addinfo['開催地'] = results_addinfo.index.map(lambda x :venue[str(x)[4:6]])
            results_addinfo = Results.scrape_preprocessing(results_addinfo)
            results_mergeinfo = pd.concat([pre_df,results_addinfo],axis=0)

            return results_mergeinfo
        

    def preprocessing(self):
        self.data_p = self.data.drop(['調教師','着差'],axis=1)
        times = self.data['タイム'].to_list()
        self.data_p['タイム'] = [float(time.split(':')[0])*60+float(time.split(':')[1]) for time in times ]
        self.data_p['entropy'] = round(-np.log2(1/(self.data_p['単勝']/0.7)),2)
        self.data_p['course_len'] = self.data_p['course_len'].map(lambda x: int(x))
        self.data_p['y__binary_in3'] = self.data_p['着順'].map(lambda x: 1 if x <4 else 0)
        self.data_p['y__binary_in1'] = self.data_p['着順'].map(lambda x: 1 if x <2 else 0)
        self.data_p['horse_len'] = self.data_p.groupby(level=0)['着順'].max()
        self.data_p['rank_per'] = self.data_p['着順']/self.data_p['horse_len']
        self.data_p['y__multi_3'] = self.data_p['rank_per'].map(lambda x:0 if x <0.33 else (1 if x < 0.66 else 2) )
        self.data_p['y__multi_5'] = self.data_p['rank_per'].map(lambda x: 0 if x <0.2 else (1 if x < 0.4 else (2 if x < 0.6 else (3 if x < 0.8 else 4 ) ) ))
        self.data_p['y__reg_着順'] = round(self.data_p['rank_per'],2)
        self.data_p['y__reg_entropy'] = self.data_p['entropy']




