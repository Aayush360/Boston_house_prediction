# Boston_house_prediction

I have used load_boston() dataset from sklearn for predicting the price of house and Flask for serving locally. This model has R-squared value of 0.792 and Mean 
Squared Error of 0.035. 

I have dropped AGE and INDUS feature as they donot give much explanatory power to the model. 
For the target PRICE I used log_transformation to make the model linear. 
