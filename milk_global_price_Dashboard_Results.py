# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:48:29 2023

@author: gusta
"""

#DASHBOARD and RESULTS

from flask import Flask, render_template
import matplotlib.pyplot as plt
import io
import base64
import os 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


#directory paths  
general_projects_path = r"C:\Users\gusta\Desktop\Personal_Projects"

#data paths 
datasets_path = general_projects_path + r"\Datasets"
global_milk_path = datasets_path + r"\Global_Milk"
inale_data_path = global_milk_path + r"\INALE"
warehouse_path = datasets_path + r"\Data_Warehouse"
#codes paths 
milk_project_path = general_projects_path + r"\Global_Milk"
models_path = milk_project_path + r"\models_saved"
#final results path
final_results = milk_project_path + r'\final_output'

df_final = pd.read_csv(os.path.join(final_results, 'test_results.csv'), sep = ';', decimal = '.')
df_pred = pd.read_csv(os.path.join(final_results, 'predictions.csv'), sep = ';', decimal = '.')

date_col = 'Date'
target_col = 'Whole Milk Powder'
app = Flask(__name__)

@app.route('/')
def dashboard():
    # Code to generate your dashboard content
    # Create a bar plot
    x = df_final[date_col]
    y = df_final['Metric_ACC']
    y_pred = df_final['prediction_label']
    y_real = df_final[target_col]
    y_future = df_pred['prediction_label']
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    ax1.bar(x, y, color='blue', label='Accuracy Over MAPE')
    ax1.set_xlabel(date_col)
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(date_col, rotation=35)
    ax1.legend(loc='upper left')
    
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
    plt.subplots_adjust(bottom=0.3)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_data = base64.b64encode(img.read()).decode()
    
    
    #Second Plot
    fig, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(x, y_future, label='Predictions for next Auctions', marker='o')
    ax3.set_xticklabels(x, rotation=35)
    ax3.set_ylabel('Global Milk Prices', color='black')
    plt.title('Predictions for next Auctions')
    plt.tight_layout()

    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    plt.close(fig)
    img2.seek(0)
    plot_data2 = base64.b64encode(img2.read()).decode()
    # Show the plot
    plt.tight_layout()

    data = {
        'Date':x,
        'Y Real': y_real,
        'Y Pred': y_pred.round(1),
        'Acc': y
    }
    df = pd.DataFrame(data)

    # Convert DataFrame to HTML table
    table_html = df.to_html(classes='table table-bordered')
    
    data2 = {
        'Date':df_pred[date_col],
        'Prediction': df_pred['prediction_label'].round(1)
        
    }
    df2 = pd.DataFrame(data2)

    # Convert DataFrame to HTML table
    table_html2 = df2.to_html(classes='table table-bordered')
    # Return the rendered HTML template
    return render_template("milk_global_dash_html.html", plot_data=plot_data,  plot_data2=plot_data2, 
                           table_html=table_html, table_html2=table_html2)



if __name__ == '__main__':
    app.run(debug=True)