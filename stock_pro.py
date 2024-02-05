#Realoding the new file and setting the Month column as datetime and index.
df = pd.read_csv('data/New_ Consumption_values.csv')
#File was saved with: new_df.to_csv('New_ Consumption_values.csv')
df['Month'] = pd.to_datetime(df['Month'])
#df = df.set_index('Month')
#Decting and excluding outliers using the IQR strategy 
def remove_outliers(df, column):
 q1 = df[column].quantile(0.25)
 q3 = df[column].quantile(0.75)
 iqr = q3 - q1
 upper_boundary = q3 + 1.5 * iqr
 lower_boundary = q1 - 1.5 * iqr
 new_df = df.loc[(df[column] > lower_boundary) & (df[column] < upper_boundary)]
 return new_df
df = remove_outliers(df,'Total usage value')
df.shape
from statsmodels.tsa.stattools import adfuller
ADF_result = adfuller(df['total_usage_value'])
#Retrieve the ADF statistic, which is the first value in the list of results
print(f'ADF Statistic: {ADF_result[0]}') 
#Retrieve the p-value, which is the second value in the list of results
print(f'p-value: {ADF_result[1]}')
# formatting the datatypes for transformation
df['month'] = pd.to_datetime(df['month'])
#applying the transform
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method ='yeo-johnson')
df_stock['xform_values'] =pt.fit_transform(df_stock['total_usage_value'].values.reshape(-1,1))
f, ax = plt.subplots(2, 1)
df['Total usage value'].plot.hist(ax=ax[0], title='Total usage value', bins=50)
df['xform_values'].plot.hist(ax=ax[1], title='xform_values', bins=50)
plt.tight_layout()
#saving the transformed data set as ‘total_usage_value.csv'
          df_clean.to_csv(total_usage_value.csv')


#Splitting data for training and testing
idx = round(len(df_stock) * 0.90)
train = df_stock[:idx]
test = df_stock[idx:]
print(f'Train: {train.shape}')
print(f'Test: {test.shape}')
from prophet import Prophet
model = Prophet().fit(train) 
Future = model.make_future_dataframe(periods = 120, freq='M')
forecast = model.predict(Future)
#first covert to data frame
column_names =['ds','yhat','yhat_lower','yhat_upper']
forecast_df = pd.DataFrame(forecast,columns = column_names)
from prophet import Prophet
model = Prophet().fit(train) 
# Since the data points are monthly, we use a frequency = M and period of forecast is 12 *10
Future = model.make_future_dataframe(periods = 120, freq='M')
#Coverting the forecast to pandas Dataframe.
column_names =['ds','trend','trend_lower','trend_upper','yhat','yhat_lower','yhat_upper']
forecast_df = forecast_df = pd.DataFrame(forecast,columns = column_names) #Inverse transforming
forecast_df[['yhat']]= pt.inverse_transform(forecast[['yhat']])
forecast_df[['trend']]= pt.inverse_transform(forecast[['trend']])
forecast_df[['trend_lower']]= pt.inverse_transform(forecast[['trend_lower']])
forecast_df[['trend_upper']]= pt.inverse_transform(forecast[['trend_upper']])
forecast_df[['yhat_lower']]= pt.inverse_transform(forecast[['yhat_lower']])
forecast_df[['yhat_upper']]= pt.inverse_transform(forecast[['yhat_upper']])
#first covert to data frame
column_names =['ds','trend','trend_lower','trend_upper','yhat','yhat_lower','yhat_upper']
forecast_df = pd.DataFrame(forecast_df,columns = column_names)#plotting the forecast
Fig = model.plot(forecast_df )
Plot.show()
#plot changepoints
from prophet.plot import add_changepoints_to_plot
fig = model.plot(forecast_df,ylabel='Fiel OperationsStock Consumption')
add_changepoints_to_plot(fig.gca(), model, forecast_df)
plt.show()
#model evaluation

import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Perform cross-validation
df_cv = cross_validation(model, initial=' 3650 days', period='180 days',horizon='365 days')
# Calculate performance metrics
df_metrics = performance_metrics(df_cv)
# Calculate MAE, MSE, and RMSE
mae = mean_absolute_error(df_cv['y'], df_cv['yhat'])
mse = mean_squared_error(df_cv['y'], df_cv['yhat'])
rmse = np.sqrt(mse)
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')

#hyperparatmeter tuning
# define our parameter grid
param_grid = {'changepoint_prior_scale': [ 0.1, 0.01]
 'seasonality_prior_scale': [1.0, 0.1,]
 'seasonality_mode': ['additive',
 'multiplicative']}
#create an empty list to hold all of the RMSE values, assuming that’s our chosen performance metric:
import numpy as np
import itertools
all_params = [dict(zip(param_grid.keys(), value))
 for value in itertools.product(*param_grid.values())]
rmse_values= []
#cutoffs
cutoffs = [pd.Timestamp('{}-{}-{}'.format(year, month,
 day))
 for year in range(2010, 2019)
 for month in range(1, 13)
 for day in [1, 11, 21]]
for params in all_params:
 model = Prophet(yearly_seasonality=4, **params).fit(forecast_df)
 df_cv = cross_validation(model,
 cutoffs=cutoffs,
horizon='360 days',
parallel='processes')
 df_p = performance_metrics(df_cv, rolling_window=1)
 rmse_values.append(df_p['rmse'].values[0])

