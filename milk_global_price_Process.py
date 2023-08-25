# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 08:41:05 2023

@author: gusta
"""


import os 
import matplotlib
import warnings
warnings.filterwarnings("ignore")


#directory paths  
general_projects_path = r"C:\Users\gusta\Desktop\Personal_Projects"

#data paths 
milk_project_path = general_projects_path + r"\Global_Milk"
datasets_path = milk_project_path + r"\Global_Milk_Data"
inale_data_path = datasets_path + r"\INALE"
warehouse_path = datasets_path + r"\Data_Warehouse"

#codes paths  
models_path = milk_project_path + r"\models_saved"

#outputs paths  
final_results = milk_project_path + r'\final_output'

plot_path = milk_project_path + r'\plots_saved'
plot_best_model_path = plot_path + r'\best_model'
plot_tuned_model_path = plot_path + r'\tuned_model'
plot_test_data_path = plot_path + r'\test_data'
plot_future_data_path = plot_path + r'\future_data'
os.chdir(milk_project_path)

import milk_global_price_DATA_SCRAP_AND_PREP as download_code
#The python file below use matplotlib. In order to avoid plots blocking the script
#set the 'dont_plot_for_script' equal True
dont_plot_for_script = True
if dont_plot_for_script == True:
    matplotlib.use('Agg') 
import milk_global_price_PyCaret_Analysis as data_code

#Websites
inale_link = "https://www.inale.org/estadisticas/"

#configs
download_data = False
target_col = 'Whole Milk Powder'
date_col = 'Date'
max_lag = 12
obs_eval = 6
num_iter = 30

 #Check all options on https://pycaret.readthedocs.io/en/stable/api/regression.html#pycaret.regression.setup
def data_download():
    
    download_code.download_milk_data(  inale_data_path, inale_link)
    download_code.data_pred_milk(inale_data_path, warehouse_path)
    
    return 'data downloaded'

def data_prep_and_regression():

    df_final, df_not_realized, regression_results = data_code.pycaret_template(target_col, max_lag, obs_eval, date_col, num_iter,
                                                           models_path, plot_best_model_path, plot_tuned_model_path,
                                                           plot_test_data_path) 
    
    df_pred = data_code.load_and_predict(df_not_realized, date_col, models_path, plot_future_data_path)

    return df_final, df_pred, regression_results


if __name__ == "__main__":
    
    if download_data == True:
        print('Data will be downloaded')
        data_download()
    else:
        print('Data not downloaded')
        
        
    df_final, df_pred, models_results = data_prep_and_regression()
    models_results.to_csv(os.path.join(final_results,'models_metrics.csv'), sep = ';', decimal = '.')
    df_final.to_csv(os.path.join(final_results,'test_results.csv'), sep = ';', decimal = '.')
    df_pred.to_csv(os.path.join(final_results,'predictions.csv'), sep = ';', decimal = '.')