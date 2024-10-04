
import pandas as pd
import tqdm
import re
import time
import requests
from bs4 import BeautifulSoup
from Class.ScrapedDataProcessor import ScrapedDataProcessor


from selenium.webdriver import Chrome,ChromeOptions
from selenium.webdriver.common.by import By

venue = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}


class ShutubaTable(ScrapedDataProcessor):
    def __init__(self,shutuba_tables):
        super(ShutubaTable,self).__init__()
        self.data = shutuba_tables


    @classmethod
    def scrape(clf,race_id_list,date):
        data= pd.DataFrame()
        for race_id in tqdm.tqdm(race_id_list):
            url = 'https://race.netkeiba.com/race/shutuba.html?race_id='+race_id
            df = pd.read_html(url,encoding="shift_jis")[0]
            df = df.T.reset_index(level=0,drop=True).T

            html = requests.get(url)
            html.encoding = 'EUC-JP'
            soup = BeautifulSoup(html.text, 'html.parser')
            
            texts = soup.find('div',attrs={'class':'RaceData01'}).text
            texts = re.findall(r'\w+',texts)
            for text in texts:
                if 'm' in text:
                    df['course_len'] = [int(re.findall(r'\d+',text)[0])]*len(df)
                if text in ['曇','晴','雨','小雨','小雪','雪']:
                    df['weather'] = [text] * len(df)
                if text in ['良','稍重','重']:
                    df['ground_state'] = [text] * len(df)
                if '不' in text:
                    df['ground_state'] = ['不良'] * len(df)
                if '芝' in text:
                    df['race_type'] = ['芝'] * len(df)
                if '障' in text:
                    df['race_type'] = ['障害'] * len(df)
                if 'ダ' in text:
                    df['race_type'] = ['ダート'] * len(df)
            df['date'] = [date] * len(df)

            horse_id_list = []
            horse_td_list = soup.find_all('td',attrs={'class':'HorseInfo'})
            for td in horse_td_list:
                horse_id = re.findall(r'\d+',td.find('a')['href'])[0]
                horse_id_list.append(horse_id)

            jockey_id_list = []
            jockey_td_list = soup.find_all('td',attrs={'class':'Jockey'})
            for td in jockey_td_list:
                jockey_id = re.findall(r'\d+',td.find('a')['href'])[0]
                jockey_id_list.append(jockey_id)

            df['horse_id'] = horse_id_list
            df['jockey_id'] = jockey_id_list

            df.index = [race_id] * len(df)
            data = pd.concat([data, df])
            time.sleep(1)
        return clf(data)

    def preprocessing(self): 
        df = self.data.copy()
        #性齢を性別と年齢に分ける
        df["性"] = df["性齢"].map(lambda x :str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x :str(x)[1:]).astype(int)

        #馬体重を体重と体重変化に分ける
        df = df[df['馬体重 (増減)'] != '--']
        df['体重'] = df[df['馬体重 (増減)'].notna()]['馬体重 (増減)'].map(lambda x : int(str(x).split('(')[0]))
        df['体重変化'] = df[df['馬体重 (増減)'].notna()]['馬体重 (増減)'].map(lambda x : int(str(x).split('(')[1][:-1]))

        df['馬番'] = df['馬 番']
        df['date'] = pd.to_datetime(df['date'])

        df['R'] = df.index.map(lambda x :str(x)[-2:])
        df['corse_used_num']= df.index.map(lambda x :int(str(x)[6:8]))
        df['開催地'] = df.index.map(lambda x :venue[str(x)[4:6]])



        df = df[['枠', '馬番', "性",'年齢', '斤量', '体重','体重変化', 'course_len', 'race_type',
        'weather', 'ground_state', 'date', 'R','corse_used_num','開催地','horse_id', 'jockey_id']]
        df = df.rename(columns={'枠':'枠番','馬 番':'馬番'})
        df['馬番']= df['馬番'].astype(int)
        df['枠番']= df['枠番'].astype(int)
        df['斤量']= df['斤量'].astype(int)
        self.data_p =df

    @classmethod      
    def scrape_shutuba_table_ChromeDriver(self,race_id_list,date):
        options = ChromeOptions()
        driver = Chrome(options=options)
        shutuba_table = pd.DataFrame([])


        for race_id in tqdm.tqdm(race_id_list):
            df = pd.DataFrame([])
            url = 'https://race.netkeiba.com/race/shutuba.html?race_id='+race_id

            driver.get(url)
            elements = driver.find_elements(By.CLASS_NAME,'HorseList')
            for element in elements:
                tds = element.find_elements(By.TAG_NAME,'td')
                row = []
                for td in tds:
                    row.append(td.text)
                    if td.get_attribute('class') in ['HorseInfo','Jockey']:
                        if td.find_elements(By.TAG_NAME,'a'):
                            href = td.find_elements(By.TAG_NAME,'a')[0].get_attribute('href')
                            row.append(re.findall(r'\d+',href)[0])
                        else:
                            row.append("")
                df = pd.concat([df,pd.DataFrame([row],index=[race_id])]) #self.shutuba_table.append(pd.Series(row,name=race_id))
            
            df = self.preprocessing_chrome(df)
            elements = driver.find_elements(By.CLASS_NAME,'RaceData01')
            text = elements[0].text
            # コースの長さを抽出
            course_len_match = re.search(r'(\d+)m', text)
            if course_len_match:
                df['course_len'] = [int(course_len_match.group(1))] * len(df)

            # 天候を抽出
            weather_match = re.search(r'天候:(\S+)', text)
            if weather_match:
                weather = weather_match.group(1)
                if weather in ['曇','晴','雨','小雨','小雪','雪']:
                    df['weather'] = [weather] * len(df)

            # 馬場状態を抽出
            ground_state_match = re.search(r'馬場:(\S+)', text)
            if ground_state_match:
                ground_state = ground_state_match.group(1)
                if ground_state in ['良', '稍重', '重', '不良']:
                    df['ground_state'] = [ground_state] * len(df)

            # レースタイプを抽出
            if '芝' in text:
                df['race_type'] = ['芝'] * len(df)
            elif '障' in text:
                df['race_type'] = ['障害'] * len(df)
            elif 'ダ' in text:
                df['race_type'] = ['ダート'] * len(df)
            df['date'] = date

            shutuba_table = pd.concat([shutuba_table,df])
        driver.close()
        return shutuba_table

    @classmethod 
    def preprocessing_chrome(self,data):
            #性齢を性別と年齢に分ける
        df = data.copy()
        df = df[[0,1,3,4,5,6,7,8,10,11,12]]
        df.columns = ['枠番','馬番','馬名','horse_id','性齢','斤量','騎手','jockey_id','馬体重','単勝','人気']
        
        #性齢を性別と年齢に分ける
        df["性"] = df["性齢"].map(lambda x :str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x :str(x)[1:]).astype(int)

        #馬体重を体重と体重変化に分ける
        df['体重_num'] = pd.to_numeric(df['馬体重'].str.split('(', expand=True)[0], errors='coerce')
        # 数値がある行のみを残す
        df = df.dropna(subset=['体重_num'])
        # 必要ならば、元の列名に戻す
        df['体重'] = df['体重_num']
        df = df.drop(columns=['体重_num'])
        df['体重変化'] = df['馬体重'].str.split('(',expand=True)[1].str[:-1].astype(int)
        
        #データをint,floatに変換
        df["単勝"] = df["単勝"].astype(float)

        df['R'] = df.index.map(lambda x :str(x)[-2:])
        df['corse_used_num']= df.index.map(lambda x :int(str(x)[6:8]))
        df['開催地'] = df.index.map(lambda x :venue[str(x)[4:6]])

        df = df[['枠番','馬番','馬名','horse_id','性','斤量','騎手','jockey_id','体重','単勝','人気',"年齢",'体重変化','R','開催地']]
        self.data_p = df
        return df
    

