
#  notebook imports

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# gather data

boston_data = load_boston()
data = pd.DataFrame(data=boston_data.data, columns=boston_data.feature_names)
features = data.drop(['INDUS','AGE'],axis=1)

log_prices = np.log(boston_data.target)
target = pd.DataFrame(log_prices, columns=['PRICE']) # converting into 2-d from flat array


CRIM_IDX =0
ZN_IDX=1
CHAS_IDX =2
RM_IDX=4
PTRATIO_IDX=8

property_stats = np.ndarray(shape=(1,11))

property_stats = features.mean().values.reshape(1,11)



regr = LinearRegression().fit(features,target)
fitted_vals = regr.predict(features)

# calcualating mse and rmse
mse_values = mean_squared_error(target, fitted_vals)
rmse_values = np.sqrt(mse_values)


# define a function to output the log_price

def get_log_estimate(nr_rooms,
                    students_per_class,
                    next_to_river=False,
                    high_confidence=True):
    # configure property
    property_stats[0][RM_IDX]= nr_rooms
    property_stats[0][PTRATIO_IDX]= students_per_class
    if next_to_river:
        property_stats[0][CHAS_IDX]= 1
    else:
        property_stats[0][CHAS_IDX]= 0

     # make predictions
    log_estimate = regr.predict(property_stats)[0][0]
    
    # calculate range
    
    if high_confidence:
        # for 2 stdev
        upper_bound = log_estimate+2*rmse_values
        lower_bound = log_estimate-2*rmse_values
        interval = 95
    
    else:
        # for 1 stdev
        upper_bound = log_estimate+rmse_values
        lower_bound = log_estimate-rmse_values
        interval= 68
    
#     actual_price = np.e**log_estimate+25.56*(np.e**log_estimate)
        
    return log_estimate, upper_bound, lower_bound, interval
    

# pull out median price from the dataset

original_median_price = np.median(boston_data.target)


ZILLOW_MEDAIN_PRICE = 583.3
scale_factor = ZILLOW_MEDAIN_PRICE/original_median_price
log_est, u_bound, l_bound, interval = get_log_estimate(9, students_per_class=15, next_to_river=False,
                                                      high_confidence=False)

# convert to today's price (scaling and rounding)

dollar_est = round(np.e**log_est*1000*scale_factor,-3)
u_bound = round(np.e**u_bound*1000*scale_factor,-3)
l_bound = round(np.e**l_bound*1000*scale_factor,-3)


print(f'the estimated dollar value is: {dollar_est}')
print(f'At {interval}% confidence interval the valuation range for this property is: USD {l_bound} to USD {u_bound}')




def get_dollar_estimate(rm, ptratio, chas=False, large_range=True):
    """
    Estimate the price of property in Boston.
    
    parameters:
    --------------------------------------------
    
    rm: number of rooms for the property
    ptratio: student to teacher ratio in classroom
    chas: is it next to Charles River
    large_range: if you want 95% confidence interval for your prediction else gives 68%
    """
    
    if rm < 1 or ptratio < 1 :
        print('That is unrealistic. Try again!')
        return 
    
    
    log_est, u_bound, l_bound, interval = get_log_estimate(rm, students_per_class=ptratio, next_to_river=chas,
                                                      high_confidence=large_range)

    # convert to today's price (scaling and rounding)

    dollar_est = round(np.e**log_est*1000*scale_factor,-3)
    u_bound = round(np.e**u_bound*1000*scale_factor,-3)
    l_bound = round(np.e**l_bound*1000*scale_factor,-3)

    return dollar_est,u_bound,l_bound
