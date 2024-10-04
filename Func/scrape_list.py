# netkeibaからスクレイピング
# 01:札幌、02:函館,03:福島,04:新潟,05:東京,06:中山,07:中京,08:京都,09:阪神,10:小倉,

def get_scrape_list(start,end):
  venue = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
  race_id_list = []
  for year in range(start,end+1,1):
    for place in range(1,11,1):
      for kai in range(1,6,1):
        for day in range(1,9,1):
          for r in range(1,13,1):
            race_id = str(year)+str(place).zfill(2)+str(kai).zfill(2)+str(day).zfill(2)+str(r).zfill(2)
            race_id_list.append(race_id)
  return race_id_list