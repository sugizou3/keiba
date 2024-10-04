import pandas as pd
import tqdm
import numpy as np
import time


class HorseResults:
  def __init__(self,horse_results):
    self.horse_results = horse_results[['日付','着 順','賞金',]]
    self.horse_results = self.horse_results.rename(columns={'着 順':'着順'})
    

    self.preprocessing()

  @classmethod
  def read_pickle(cls,path_list):
      df = pd.concat([pd.read_pickle(path) for path in path_list]) 
      return cls(df)

  @staticmethod
  def scrape(horse_id_list,pre_df=None):
    horse_results = {}
    if pre_df is not None:
      horse_id_list = [horse_id  for horse_id in horse_id_list if not horse_id in pre_df.index.unique()]
      
    for horse_id in tqdm.tqdm(horse_id_list):
      
      try:
        url = 'https://db.netkeiba.com/horse/'+horse_id
        df = pd.read_html(url)[3]
        if not '日付' in df.columns:
          df = pd.read_html(url)[4]
        horse_results[horse_id] = df

        time.sleep(1)
      except IndexError:
        continue
      except:
        break

    for key in horse_results.keys():
      horse_results[key].index = [key]*len(horse_results[key])

    horse_results = pd.concat([horse_results[key] for key in horse_results.keys()],sort=False)
    results_mergeinfo = pd.concat([pre_df,horse_results],axis=0)

    return results_mergeinfo

  def preprocessing(self):
    df = self.horse_results.copy()
    #着順の数字以外の文字列が含まれているものを取り除く
    # df = df[~(df["着順"].astype(str).str.contains("\D"))]
    df['着順'] = pd.to_numeric(df['着順'],errors='coerce')
    df.dropna(subset=['着順'],inplace=True)
    df["着順"] = df["着順"].astype(int)

    df['date'] = pd.to_datetime(df['日付'])
    df.drop(['日付'],axis=1,inplace=True)

    df['賞金'].fillna(0,inplace=True)

    self.horse_results = df

  
  def horse_prize(self,horse_id_list,date):
    target_df = self.horse_results.loc[horse_id_list]
    filtered_df = target_df[target_df['date']==date]
    filtered_df = filtered_df.rename(columns={'賞金':'賞金_horse'})
    
    return filtered_df.fillna({'賞金_horse':0})['賞金_horse']
  
  def merge_horse_prize(self,results,date):

    df  = results[results['date'] == date]
    horse_id_list = df['horse_id']
    merged_df = df.merge(self.horse_prize(horse_id_list,date),left_on='horse_id',right_index=True,how='left')
    
    return merged_df
  
  def merge_all_horse_prize(self,results):
    date_list = results['date'].unique()
    # print('horse_prizeをmergeしています')
    merged_df = pd.concat([self.merge_horse_prize(results,date) for date in tqdm.tqdm(date_list)])
    
    return merged_df
  
  def race_horse_prize(self,results):
    horse_prize_list = self.merge_all_horse_prize(results)['賞金_horse']
    race_sum = horse_prize_list.groupby(level=0).sum().to_frame()
    race_sum = race_sum.rename(columns={'賞金_horse':'賞金_race'})

    return race_sum
  
  def merge_race_prize(self,results):
    merged_df = results.merge(self.race_horse_prize(results),left_index=True,right_index=True,how='left')
    return merged_df


  def past_race(self,horse_id_list,date,n_samples=10):
    target_df = self.horse_results.loc[horse_id_list]
    
    if n_samples > 0:
      filtered_df = target_df[target_df['date']<date]. \
        sort_values('date',ascending=False).groupby(level=0).head(n_samples)
    else:
      raise Exception('n_sample must be >0')
    
    filtered_df2= filtered_df.reset_index().sort_values('index',ascending=False)
    filtered_df3 = pd.pivot_table(filtered_df2, index='index', columns=filtered_df2.groupby('index').cumcount())
    filtered_df3.columns = ['_'.join([col[0],str(col[1])]).strip() for col in filtered_df3.columns.values]
    result_df = filtered_df3[filtered_df3.columns[['着順' in col for col in filtered_df3.columns]]]
    return result_df

  def merge_past_race(self,results,date,n_samples=10):
    
    df  = results[results['date'] == date]
    horse_id_list = df['horse_id']
    merged_df = df.merge(self.past_race(horse_id_list,date,n_samples),left_on='horse_id',right_index=True,how='left')
    
    return merged_df
  
  def merge_all_past_race(self,results,n_samples=10):
    date_list = results['date'].unique()
    # print('past_raceをmergeしています')
    merged_df = pd.concat([self.merge_past_race(results,date,n_samples) for date in tqdm.tqdm(date_list)])
    
    return merged_df


  def average(self,horse_id_list,date,n_samples='all'):
    target_df = self.horse_results.query('index in @horse_id_list')#loc[horse_id_list]
    
    if n_samples == 'all':
      filtered_df = target_df[target_df['date']<date]
    elif n_samples > 0:
      filtered_df = target_df[target_df['date']<date]. \
        sort_values('date',ascending=False).groupby(level=0).head(n_samples)
    else:
      raise Exception('n_sample must be >0')
    # filtered_df = filtered_df.rename(columns={'着順':'着順_ave'})

    average= filtered_df.groupby(level=0)[['着順','賞金']].mean()
    return average.rename(columns={'着順':'着順_{}R'.format(n_samples),'賞金':'賞金_{}R'.format(n_samples)})

  def merge_average(self,results,date,n_samples='all'):
    
    df  = results[results['date'] == date]
    horse_id_list = df['horse_id']
    merged_df = df.merge(self.average(horse_id_list,date,n_samples),left_on='horse_id',right_index=True,how='left')
    
    return merged_df

  def merge_all_average(self,results,n_samples='all'):
    date_list = results['date'].unique()
    # print('averageをmergeしています')
    # merged_df = merged_df.rename(columns={'着順':'着順_ave'})
    merged_df = pd.concat([self.merge_average(results,date,n_samples) for date in tqdm.tqdm(date_list)])
    
    return merged_df
  