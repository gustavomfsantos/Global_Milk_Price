# Global_Milk_Price
Global Milk Price ( Whole Milk Powder - Global Dairy Trade) prediction Using PyCaret. 

This project includes two main functions. A Scrap/Download code where public data is collected and preprocessed to be readable as DataFrame.
The second part is a final dataset process cleaning NAN and creating Lags from the target feature as predictor feature. After that, PyCaret is used to 
chose among various algorithms and train a model to predict Whole Milk Powder price traded in GDT biweekly auctions.

The final output will be plots showing the performance on training process, on the test set and will bring future projections, prices for the next
auctions. The paths and all the action are defined on file "milk_global_price_Process.py". Can be executed through terminal or inside any Python IDE,
such as Spyder. The file "milk_global_price_Dashboard_Results.py" is a test do deploy a local dashboard using the results from the model.

The goal of this model is to help bring predictability in global milk prices to help producers adjust their production during high prices and buyers to stock when peaks are predicted
