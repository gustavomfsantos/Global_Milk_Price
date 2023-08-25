# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:55:21 2023

@author: gusta
"""

import os 
import time
import pandas as pd
import numpy as np
import datetime
from functools import reduce

from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
#directory paths

general_projects_path = r"C:\Users\gusta\Desktop\Personal_Projects"
milk_project_path = general_projects_path + r"\Global_Milk"
#data paths
datasets_path = milk_project_path + r"\Global_Milk_Data"
inale_data_path = datasets_path + r"\INALE"
warehouse_path = datasets_path + r"\Data_Warehouse"



#Websites
inale_link = "https://www.inale.org/estadisticas/"




def download_milk_data(  inale_data_path, scrap_link):
    
    
    
    chrome_options = Options()

    chrome_options.add_argument("--window-size=1920,1080")
 
    files = os.listdir(inale_data_path)
    if len(files)>0:
        for file in files:
            os.remove(os.path.join(inale_data_path, f'{file}'))
     
    prefs = {"download.default_directory" : inale_data_path}
    chrome_options.add_experimental_option("prefs",prefs)

    driver = webdriver.Chrome( chrome_options)    
    driver.get(scrap_link)
    time.sleep(5)        
                       
    driver.find_element(by = By.XPATH, value = '//*[@id="content"]/div/div/div[3]/div[3]/div/div[4]/a[2]').click()
    time.sleep(8)
    driver.find_element(by = By.XPATH, value = '//*[@id="content"]/div/div/div[3]/div[3]/div/div[9]/a[2]').click()
    time.sleep(5)
    driver.close()
    
    return print('GDT data and IFCN data downloaded')


def data_pred_milk(inale_data_path, warehouse_path):

    files = os.listdir(inale_data_path)
    
    gdt_file = [x for x in files if 'GDT' in x]
    try:
        df_gdt = pd.read_csv(os.path.join(inale_data_path, gdt_file[0]))
    except:
        df_gdt = pd.read_excel(os.path.join(inale_data_path, gdt_file[0]))

    #manual ajusts
    df_gdt.columns = df_gdt.iloc[13]
    df_gdt = df_gdt.tail(-14)
    df_gdt = df_gdt.iloc[:, 1:15] 
    
    #Getting date column fill
    df_gdt['Año/Mes'] = df_gdt['Año/Mes'].fillna(method = 'ffill')
    df_gdt.reset_index(drop = True, inplace = True)
    
    #get index to slice dataframes
    #whole milk
    whole_milk_slice_start = 0
    whole_milk_slice_end = int(str(datetime.date.today().year)[2:]) + 9
    df_whole_milk = df_gdt.iloc[whole_milk_slice_start:whole_milk_slice_end]
    df_whole_milk.info()
    #skimmed milk
    skimmed_milk_slice_start = whole_milk_slice_end + 5
    skimmed_milk_slice_end = skimmed_milk_slice_start + 28
    df_skimmed_milk = df_gdt.iloc[skimmed_milk_slice_start:skimmed_milk_slice_end]
    for col in df_skimmed_milk.columns:
        df_skimmed_milk[col] = df_skimmed_milk[col].astype(str).str.replace('.', '')  
    df_skimmed_milk.info()
    #cheddar cheese
    chedar_cheese_slice_start = skimmed_milk_slice_end + 6
    chedar_cheese_slice_end = chedar_cheese_slice_start + 26
    df_chedar_cheese = df_gdt.iloc[chedar_cheese_slice_start:chedar_cheese_slice_end]
    df_chedar_cheese.info()
    #average dairy price
    avg_price_slice_start = chedar_cheese_slice_end + 5
    avg_price_slice_end = avg_price_slice_start + 32
    df_avg_price = df_gdt.iloc[avg_price_slice_start:avg_price_slice_end]
    df_avg_price.info()
    # # Got all 4 types of product from the file. Four diferents dfs
  
    df_list = [ df_whole_milk, df_skimmed_milk, df_chedar_cheese, df_avg_price  ]        
    
        

    df_list_final = []
    for df in df_list:
       
        df = df.melt(id_vars=["Año/Mes",'Evento'], 
          var_name="Mês", 
          value_name="Value")
        

        df.columns = ['Year', 'Day', 'Month','Value']
        print("Creating month columns..")
        choices = [df['Month']=='Ene ',df['Month']=='Feb',df['Month']=='Mar',df['Month']=='Abr',
                   df['Month']=='May',df['Month']=='Jun',df['Month']=='Jul',df['Month']=='Ago',
                   df['Month']=='Sep',df['Month']=='Oct',df['Month']=='Nov',df['Month']=='Dic']
        conditions = [x for x in range(1,13)]
        
        df['Month_aux'] = np.select(choices, conditions)
        
        df['Day_aux'] = np.where(df['Day'] == '1er', 1,15)
        
        df['Date'] = pd.to_datetime(df['Year'].astype(str) +'-'+df['Month_aux'].astype(str) + '-'+df['Day_aux'].astype(str))
        df = df.drop(['Year', 'Month','Month_aux', 'Day', 'Day_aux'], axis=1).sort_values('Date')
        
        df['Value'] = df['Value'].apply(lambda x: float(x))
        df_list_final.append(df)
        
    df_global_price = reduce(lambda left,right: pd.merge(left,right,on='Date', how ='left'),df_list_final)  
    
    df_global_price = df_global_price.drop_duplicates()
    
    df_global_price.columns = ['Whole Milk Powder', 'Date', 
                               'Skimmed Milk Powder',
                               'Chedar Cheese', 'Average Price Index']
    
    df_global_price = df_global_price[[ 'Date', 'Whole Milk Powder',
                               'Skimmed Milk Powder',
                               'Chedar Cheese', 'Average Price Index']]
    
    #The biweekly frequency started only on September of 2010. To change the monthly frequency of previous data
    # we will interpolate the values using the previusly and upcoming month values.
    df_global_price[[ 'Whole Milk Powder',
                               'Skimmed Milk Powder',]] = df_global_price[[ 'Whole Milk Powder',
                               'Skimmed Milk Powder',]].interpolate(limit = 1,
                                                                limit_direction = 'backward',
                                                                method = 'linear') 
                                                                    
    ###Next Files                                                               
    ifcn_file = [x for x in files if 'IFCN' in x]
    
    try:
        df_ifcn = pd.read_csv(os.path.join(inale_data_path, ifcn_file[0]))
    except:
        df_ifcn = pd.read_excel(os.path.join(inale_data_path, ifcn_file[0]))
    
    #manual ajusts
    df_ifcn.columns = df_ifcn.iloc[12]
    df_ifcn = df_ifcn.tail(-13)
    df_ifcn = df_ifcn.iloc[:, 1:14] 
    df_ifcn.drop(df_ifcn.tail(3).index,inplace=True)
    
    df_ifcn = df_ifcn.melt(id_vars=["Año/Mes"], 
          var_name="Mês", 
          value_name="Value")
    df_ifcn.columns = ['Year', 'Month','IFCN Price Index']
    print("Creating month columns..")
    choices = [df_ifcn['Month']=='Ene ',df_ifcn['Month']=='Feb',df_ifcn['Month']=='Mar',df_ifcn['Month']=='Abr',
               df_ifcn['Month']=='May',df_ifcn['Month']=='Jun',df_ifcn['Month']=='Jul',df_ifcn['Month']=='Ago',
               df_ifcn['Month']=='Sep',df_ifcn['Month']=='Oct',df_ifcn['Month']=='Nov',df_ifcn['Month']=='Dic']
    conditions = [x for x in range(1,13)]
    
    df_ifcn['Month_aux'] = np.select(choices, conditions)
    df_ifcn['Date'] = pd.to_datetime(df_ifcn['Year'].astype(str) +'-'+df_ifcn['Month_aux'].astype(str) + '-'+'01')
    df_ifcn = df_ifcn.drop(['Year', 'Month','Month_aux'], axis=1).sort_values('Date')
    df_ifcn.columns = ['IFCN Milk Price Index', 'Date']
    df_ifcn['IFCN Milk Price Index'] = df_ifcn['IFCN Milk Price Index'].apply(lambda x: float(x))
    
    #This dataset IFCN has month data frequency. To use as biweekly data, we will interpolate 
    #but only after the merge, because then we eill have the full dates
    

    df_final = pd.merge(df_global_price, df_ifcn, how = 'left', on = 'Date')
    
    df_final['IFCN Milk Price Index']  = df_final['IFCN Milk Price Index'].interpolate(limit = 1,
                                                                                     limit_direction = 'backward',
                                                                                     method = 'linear') 
    
    df_final.to_csv(os.path.join(warehouse_path, "Milk_Global_Prices_Dataset.csv"), sep = ';',
                    decimal = '.')
    
    return 'Done Processing Files'

if __name__ == "__main__":
    
    download_milk_data(  inale_data_path, inale_link)
    data_pred_milk(inale_data_path, warehouse_path)
