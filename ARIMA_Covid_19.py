import pandas as pd
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Data for this example collected from https://www.ecdc.europa.eu/en/covid-19/data
# import data and print some of them
data= pd.read_csv('data.csv', header= 0, index_col= 0, parse_dates= True, squeeze= True)
print(data.head())

# we keep only the data refer to country of Greece and print some of them
isGreece= data['countriesAndTerritories']== 'Greece'
dataGreece= data[isGreece]
print(dataGreece.head())

# plot corona virus cases daily from the beginning of the pandemic
dataGreece.loc['26/02/2020':'04/07/2022'].plot(y= 'cases')
pyplot.title('Daily corona virus cases from February 2020 to early July 2022 for Greece')
pyplot.grid()
pyplot.show()

# create the autocorrelation plot of data
autocorrelation_plot(dataGreece['cases'])
pyplot.title('Autocorrelation plot of cases')
pyplot.show()

# autocorrelation with histogram
plot_acf(dataGreece['cases'], lags= 50)
pyplot.title('Autocorrelation plot using statsmodels library')
pyplot.grid()
pyplot.show()
plot_pacf(dataGreece['cases'], lags= 100)
pyplot.title('Autocorrelation of a part of the series')
pyplot.grid()
pyplot.show()

# split the dataset to datasets for every year of cases and plot autocorrelations
is2020= dataGreece['year']== 2020
data2020= dataGreece[is2020]
print(data2020['cases'].head())
plot_acf(data2020['cases'], lags= 50)
pyplot.title('Autocorrelation plot of 2020 cases')
pyplot.grid()
pyplot.show()

is2021= dataGreece['year']== 2021
data2021= dataGreece[is2021]
print(data2021['cases'].head())
plot_acf(data2021['cases'], lags= 50)
pyplot.title('Autocorrelation plot of 2021 cases')
pyplot.grid()
pyplot.show()

is2022= dataGreece['year']== 2022
data2022= dataGreece[is2022]
print(data2022['cases'].head())
plot_acf(data2022['cases'], lags= 50)
pyplot.title('Autocorrelation plot of 2022 cases')
pyplot.grid()
pyplot.show()

# create the ARIMA model and fit it with data
model= ARIMA(dataGreece['cases'].values, order= (1, 1, 2))
model_fit= model.fit()
print(model_fit.summary())

# plot residual errors
residuals= pd.DataFrame(model_fit.resid)
residuals.plot(title= 'Residual errors')
pyplot.grid()
pyplot.show()
residuals.plot(kind= 'kde', title= 'Density of errors')
pyplot.grid()
pyplot.show()

# make predictions and plot them
dataGreece['forecast']= model_fit.predict(dynamic= False)
dataGreece[['cases', 'forecast']].plot()
pyplot.title('Real data vs fitted data for covid 19 cases')
pyplot.grid()
pyplot.show()

# make the same process with datasets for every year. Create a new model with the same properties
model1= ARIMA(data2020['cases'].values, order= (1, 1, 2))
model_fit1= model1.fit()
print(model_fit1.summary())
data2020['forecast']= model_fit1.predict(dynamic= False)
data2020[['cases', 'forecast']].plot()
pyplot.title('Real data vs fitted data for 2020 cases')
pyplot.grid()
pyplot.show()

model2= ARIMA(data2021['cases'].values, order= (1, 1, 2))
model_fit2= model2.fit()
print(model_fit2.summary())
data2021['forecast']= model_fit2.predict(dynamic= False)
data2021[['cases', 'forecast']].plot()
pyplot.title('Real data vs fitted data for 2021 cases')
pyplot.grid()
pyplot.show()

model3= ARIMA(data2022['cases'].values, order= (1, 1, 2))
model_fit3= model3.fit()
print(model_fit3.summary())
data2022['forecast']= model_fit3.predict(dynamic= False)
data2022[['cases', 'forecast']].plot()
pyplot.title('Real data vs fitted data for 2022 cases')
pyplot.grid()
pyplot.show()
