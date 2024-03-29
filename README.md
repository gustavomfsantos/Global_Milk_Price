# Global_Milk_Price
Global Milk Price ( Whole Milk Powder - Global Dairy Trade) prediction Using PyCaret. 

This project includes two main functions. A Scrap/Download code where public data is collected and preprocessed to be readable as DataFrame. The second part is a final dataset process cleaning NAN and creating Lags from the target feature as predictor feature. After that, PyCaret is used to  chose among various algorithms and train a model to predict Whole Milk Powder price traded in GDT biweekly auctions. The final output will be plots showing the performance on training process, on the test set and will bring future projections, prices for the next auctions. The paths and all the action are defined on file "milk_global_price_Process.py". Can be executed through terminal or inside any Python IDE, such as Spyder. The file "milk_global_price_Dashboard_Results.py" is a test do deploy a local dashboard using the results from the model.

  -	milk_global_price_Process.py defines paths used to save and load data and plots. Also defines the basic configurations for the model, such as how many periods to predict, how many lags to use, how many iterations to do when tuning the model and the Target and Date column name. In this file we execute the download and preprocessing the data and run a Pycaret pipeline that save plots generated in the training process, to check overfitting, plots and data with predictions on test set and real values, alongside with accuracy metric and also prediction for the next periods that did not happen yet.
  -	milk_global_price_DATA_SCRAP_AND_PREP.py and milk_global_price_PyCaret_Analysis.py are python files with functions imported to the main (milk_global_price_Process.py) file to execute the project.
  -	requirements.txt contains the packages and versions needed for this project.
    
The goal of this model is to help bring predictability in global milk prices to help producers adjust their production during high prices and buyers to stock when peaks are predicted

![Final Output with Metrics and Forecast values](final_output/Dashboard_print.png)
