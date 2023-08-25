# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 17:28:46 2023

@author: gusta
"""

import os 
import pandas as pd
import numpy as np
import datetime

from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pycaret.regression import *

#directory path  
general_projects_path = r"C:\Users\gusta\Desktop\Personal_Projects"

#data paths 
milk_project_path = general_projects_path + r"\Global_Milk"
datasets_path = milk_project_path + r"\Global_Milk_Data"
inale_data_path = datasets_path + r"\INALE"
warehouse_path = datasets_path + r"\Data_Warehouse"

#codes paths 
models_path = milk_project_path + r"\models_saved"
plot_path = milk_project_path + r'\plots_saved'
plot_best_model_path = plot_path + r'\best_model'
plot_tuned_model_path = plot_path + r'\tuned_model'
plot_test_data_path = plot_path + r'\test_data'
plot_future_data_path = plot_path + r'\future_data'


os.chdir(milk_project_path)

import memory_aux

def create_dates_biweekly(start, end, date_col):
    
    start_date = start
    end_date = end
    date_range = pd.date_range(start_date, end_date)

    # Filter dates to include only days 1 and 15
    filtered_dates = date_range[date_range.day.isin([1, 15])]

    # Create a DataFrame
    df_cal = pd.DataFrame({'Date': filtered_dates})
    
    df_cal['Year'] = df_cal[date_col].dt.year
    df_cal['Month'] = df_cal[date_col].dt.month
    df_cal['Day'] = df_cal['Date'].dt.day
    # Create dummy columns for each month and year
    month_dummies = pd.get_dummies(df_cal['Month'], prefix='Month')
    year_dummies = pd.get_dummies(df_cal['Year'], prefix='Year')
    day_dummies = pd.get_dummies(df_cal['Day'], prefix='Day')
    # Concatenate the original DataFrame with the dummy columns
    df_cal = pd.concat([df_cal, month_dummies, year_dummies, day_dummies], axis=1)

    return df_cal



def data_prep_for_pycaret(max_lag, target_col, date_col):
    df = pd.read_csv(os.path.join(warehouse_path, "Milk_Global_Prices_Dataset.csv"), sep = ';',  
                     decimal = '.', index_col=0) 
    
    #remove data before 2012 
    df = df[df['Date'] >= '2012-01-01']
    
    df = df.dropna(subset = target_col, axis = 0)
    df.reset_index(drop = True, inplace = True)
    df[date_col] = pd.to_datetime(df[date_col])
    
   
    ### Get Only Target Column
    df = df[[date_col, target_col]]
    

    # Extract year and month
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    # Create dummy columns for each month and year
    month_dummies = pd.get_dummies(df['Month'], prefix='Month')
    year_dummies = pd.get_dummies(df['Year'], prefix='Year')    
    day_dummies = pd.get_dummies(df['Day'], prefix='Day')
    # Concatenate the original DataFrame with the dummy columns
    df = pd.concat([df, month_dummies, year_dummies, day_dummies], axis=1)

    # Append Future Dates on Dataset
    df_cal = create_dates_biweekly(datetime.date.today(), '2025-12-31', date_col )
    
    df = pd.concat([df, df_cal], axis=0)
    df.reset_index(drop = True, inplace = True)
    column_to_exclude = target_col
    del df_cal
    # Fill NaN values in all columns except the specified column
    fill_values = {col: 0 if col != column_to_exclude else np.nan for col in df.columns}
    df = df.fillna(fill_values)

    # Drop columns
    df.drop([date_col, 'Year', 'Month', 'Day'], axis = 1, inplace = True)
    
    # Create  Lagged target - Is taking past data to future, no leakage problem
    for lag in range(1,max_lag+1) :
        df[f'{target_col}_{lag}'] = df[f'{target_col}'].shift(lag)
    
    df = df.tail(len(df) - (max_lag +1))
    
    memory_aux.get_memory_usage()
    df = memory_aux.reduce_mem_usage(df)
    memory_aux.get_memory_usage()
    
    return df

def sep_unseen_data(df, obs_eval, max_lag, target_col):
    
    unseen_obs = obs_eval
    
    df_realized = df.dropna(subset = target_col)
    df_realized_unseen = df_realized.tail(unseen_obs)
    df_to_use =  df_realized.drop(df_realized.tail(unseen_obs).index)

    df_not_realized = df[df[target_col].isna()]
    df_not_realized = df_not_realized.drop([target_col], axis = 1)
    df_not_realized = df_not_realized.head(unseen_obs)
    
    
    #check Columns with no NAN.
    columns_not_realized = list(df_not_realized.dropna(axis = 1).columns)
    #All dfs must have same Columns
    df_to_use = df_to_use[columns_not_realized + [target_col]]
    df_realized_unseen = df_realized_unseen[columns_not_realized + [target_col]]
    df_not_realized = df_not_realized[columns_not_realized ]
    
    #drop constants?
    
    
    return df_to_use, df_realized_unseen, df_not_realized

def clean_plots_files(folder_path):
    files = os.listdir(folder_path)
    if len(files)>0:
        for file in files:
            os.remove(os.path.join(folder_path, f'{file}'))
    
    return 'Folder cleaned'

def pycaret_template(target_col, max_lag, obs_eval, date_col, num_iter,
                     models_path, plot_best_model_path, plot_tuned_model_path,
                     plot_test_data_path):
    
   

    df = data_prep_for_pycaret(max_lag, target_col, date_col)

    df, df_unseen, df_not_realized = sep_unseen_data(df, obs_eval, max_lag, target_col)
    
    # Step 3: Set up the regression enviroment. Here it can be set many things, including parallelization
    #Check all options on https://pycaret.readthedocs.io/en/stable/api/regression.html#pycaret.regression.setup
    #check log options to deploy server?
    reg_experiment = setup(df, target=target_col, session_id=123,
                           train_size = 0.8,
                           preprocess = False, 
                           # categorical_features = ['year', 'month', 'day_month'],
                           imputation_type = 'simple', numeric_imputation = 'knn',
                           normalize = 'True', normalize_method = 'minmax', 
                           data_split_shuffle = False, fold_strategy = 'timeseries', 
                           fold = 5,
                           # use_gpu = True, verbose = False,
                           # profile = True, 

                           html = False)
    
   
    
    best_model = compare_models(include = ['lr', 'en', 'br', 'huber', 'svm', 'knn', 'rf',
                                 'ada', 'gbr'], errors="raise",
                       verbose = True, 
                       n_select = 1, turbo = True, sort = 'MAPE'
                                      )
    
    #Clear plot Folder
    clean_plots_files(plot_best_model_path)
    plot_model(best_model, plot = 'error', save = plot_best_model_path)
    plot_model(best_model, plot='feature', save = (plot_best_model_path))
    plot_model(best_model, plot = 'residuals', save = (plot_best_model_path))
    
    regression_results = pull()
    df_regression_results = regression_results.round(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_regression_results.values, colLabels=df_regression_results.columns, 
             cellLoc='center', loc='center').scale(1.5, 1.5)#.set_fontsize(18)

    
    # Save the visualization as an image
    clean_plots_files(plot_test_data_path)
    plt.savefig(os.path.join(plot_test_data_path,'Dataframe_Models_Metrics.png'), bbox_inches='tight', pad_inches=0.5, dpi=300)
    plt.show()

    #Count time here
    tuned_model = tune_model(best_model, n_iter = num_iter,
                             optimize = 'MAPE', search_library = 'scikit-learn',
                             search_algorithm = 'random', return_train_score = True,
                             
                             ) 
    
    
   
    # plot_model(tuned_model)
    clean_plots_files(plot_tuned_model_path)
    plot_model(tuned_model, plot = 'error', save = (plot_tuned_model_path))
    plot_model(tuned_model, plot= 'feature', save = (plot_tuned_model_path))
    plot_model(tuned_model, plot = 'residuals', save = (plot_tuned_model_path))
    
    #finalize model train in thw whole dataset to put the model in production
    final_model = finalize_model(tuned_model)
    
    unseen_predictions = predict_model(final_model, data = df_unseen)  
    unseen_predictions['residuals'] = unseen_predictions[target_col] - unseen_predictions['prediction_label']
    
    # Create a residual plot for test data
    plt.figure(figsize=(8, 6))
    plt.scatter(unseen_predictions['prediction_label'],unseen_predictions['residuals'], alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot for Unseen Data')
    plt.grid(True)
    plt.savefig(os.path.join(plot_test_data_path,'Residual_Unseen_Data.png'))
    plt.show()
    #OK
    
    plt.figure(figsize=(8, 6))
    plt.scatter(unseen_predictions['prediction_label'], unseen_predictions['residuals'], alpha=0.7, label='Prediction Errors')
    plt.plot(np.unique(unseen_predictions['prediction_label']), np.poly1d(np.polyfit(unseen_predictions['prediction_label'],
            unseen_predictions['residuals'], 1))(np.unique(unseen_predictions['prediction_label'])),
             color='red', linestyle='--', label='Best-Fit Line')
    plt.plot(np.unique(unseen_predictions['prediction_label']), 
             np.zeros_like(np.unique(unseen_predictions['prediction_label'])),
             color='green', linestyle=':', label='Identity Line')
    plt.xlabel('Predicted Values (y_pred)')
    plt.ylabel('Prediction Errors')
    plt.title('Prediction Errors and Best-Fit Line for Unseen Data')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_test_data_path,'Best_Fit_Residual_Unseen_Data.png'))
    plt.show()
    #OK
    
    ##Podemos criar graficos do dashboard aqui
    
    #Get Metrics On unseen Data
    
    MAPE_unseen_data = mean_absolute_percentage_error(y_true = unseen_predictions['Whole Milk Powder'],
                                       y_pred = unseen_predictions['prediction_label'] )
    
    print('MAPE on unseen Data', MAPE_unseen_data)
    print('Acc on unseen Data', 1-MAPE_unseen_data)
    # Step 8: Save the model
    # You can save the trained model using save_model() function
    today = datetime.date.today()
    os.chdir(models_path)
    save_model(final_model,f'Final_Model_{today}')
    
    day = unseen_predictions[[x for x in unseen_predictions.columns if 'Day' in x]].idxmax(axis=1).str.replace('Day_', '').astype(int)
    month = unseen_predictions[[x for x in unseen_predictions.columns if 'Month' in x]].idxmax(axis=1).str.replace('Month_', '').astype(int)
    year = unseen_predictions[[x for x in unseen_predictions.columns if 'Year' in x]].idxmax(axis=1).str.replace('Year_', '').astype(int)

    # Create date column using dummy columns
    unseen_predictions['Date'] = pd.to_datetime({'year': year, 'month': month, 'day': day})
    
    
    df_final = unseen_predictions[['Date' , target_col, 'prediction_label',]]
    df_final['Metric_MAPE'] = (abs(df_final[target_col] - df_final['prediction_label']))/df_final[target_col]
    df_final['Metric_ACC'] = 1 - df_final['Metric_MAPE'] 
    
    x = df_final[date_col]
    y = df_final['Metric_ACC']
    y_pred = df_final['prediction_label']
    y_real = df_final[target_col]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    ax1.bar(x, y, color='blue', label='Accuracy Over MAPE', width=2.5)
    ax1.set_xlabel(date_col)
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    # ax1.set_xticklabels(x, rotation=35)
    ax1.legend(loc='lower left')
    
    # Create a secondary axes for the line plots
    ax2 = ax1.twinx()
    
    # Line plots
    ax2.plot(x, y_pred, color='red', label='Prediction', marker='o')
    ax2.plot(x, y_real, color='green', label='Real Value', marker='s')
    ax2.set_ylabel('Global Milk Prices', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='upper right')
    
    # Title
    plt.title('Accuracy and Global Milk Prices and Predictions')
    # plt.subplots_adjust(bottom=0.3)
    # plt.tight_layout()
    plt.savefig(os.path.join(plot_test_data_path,'Accuracy_and_Pred_and_Real_Values.png'))
    plt.show()
    
    print("Model Trained and Saved")
    df_final = memory_aux.reduce_mem_usage(df_final)
    df_not_realized = memory_aux.reduce_mem_usage(df_not_realized)
    return df_final, df_not_realized, regression_results
        
    
    
def load_and_predict(df_not_realized, date_col, models_path, plot_future_data_path):

    models_saved_list = os.listdir(models_path)
    model_file = os.path.splitext(models_saved_list[0])[0]
    saved_final_model = load_model(os.path.join(models_path, model_file))

    df_not_realized.dropna(axis = 1, inplace = True)
    new_prediction = predict_model(saved_final_model, data=df_not_realized)
    
    day = new_prediction[[x for x in new_prediction.columns if 'Day' in x]].idxmax(axis=1).str.replace('Day_', '').astype(int)
    month = new_prediction[[x for x in new_prediction.columns if 'Month' in x]].idxmax(axis=1).str.replace('Month_', '').astype(int)
    year = new_prediction[[x for x in new_prediction.columns if 'Year' in x]].idxmax(axis=1).str.replace('Year_', '').astype(int)

    # Create date column using dummy columns
    new_prediction[date_col] = pd.to_datetime({'year': year, 'month': month, 'day': day})
        
    new_prediction = new_prediction[[date_col,  'prediction_label']]
    new_prediction = memory_aux.reduce_mem_usage(new_prediction)
    
    x = new_prediction[date_col]
    y_future = new_prediction['prediction_label']
    
    clean_plots_files(plot_future_data_path)
    fig, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(x, y_future, label='Predictions for next Auctions', marker='o')
    # ax3.set_xticklabels(x, rotation=35)
    ax3.set_ylabel('Global Milk Prices', color='black')
    plt.title('Predictions for next Auctions')
    plt.savefig(os.path.join(plot_future_data_path,'Future_Predictions.png'))
    plt.show()  
    return new_prediction
    
    
if __name__ == "__main__":
    
    df_final, df_not_realized = pycaret_template(target_col, max_lag, obs_eval,date_col, num_iter,
                                                 plot_best_model_path, plot_tuned_model_path,
                                                 plot_test_data_path) 
    df_pred = load_and_predict(df_not_realized, date_col, plot_future_data_path)
