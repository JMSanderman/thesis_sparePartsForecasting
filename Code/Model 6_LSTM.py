# Imports
import pandas as pd
import numpy as np
import warnings
import pickle
from dateutil import rrule
import math
import datetime
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error
import gc
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import ast


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
pd.options.mode.chained_assignment = None  # default='warn'

#######################################################################################
#                             1. Read and Preprocess Data                             #
#######################################################################################
# load dictionary with demand data frames
with open('Data//dict_demandPublished.pkl', 'rb') as f:
    dict_demand = pickle.load(f)

# load data frame with lead time and price information
df_price_leadtime = pd.read_csv('Data//df_price_leadtimePublished.csv')

# put all spare parts that are forecasted in a list
parts = pd.read_csv('Data//df_consideredSpares.csv')
parts = parts[parts.columns[0]].values.tolist() # convert to list


#######################################################################################
#                               2. Define functions needed                            #
#######################################################################################
def prepare_data(data, window_size, forecast_size):
    # Generate the input and output sequences
    X, y = [], []
    for i in range(len(data)-window_size-forecast_size+1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+forecast_size])
    return np.array(X), np.array(y)

def test_data_preperation(data, window_size):
    # Generate the input and output sequences
    X= []
    for i in range(len(data)-window_size+1): # no minus forecast_size since we also have to include the last periods. and then we will make the forecast also using these values and compare it ourself with the actuals
        X.append(data[i:i+window_size])
    return np.array(X)


#######################################################################################
#                                    3. Define batches                                #
#######################################################################################
'''
The spare parts are distributed in batches. This is done to free up memory and distribute the 
batches across multiple computers to speed up running time.
'''
batch_size = 50
batches = []

# Loop through the IDs and split them into batches
for i in range(0, len(parts), batch_size):
    batch = parts[i:i+batch_size]
    batches.append(batch)

# Print the number of batches created
print(len(batches))


# define high level batches so allocate the batches over the different computers
highLevel_batch1 = batches[:8]
highLevel_batch2 = batches[8:16]
highLevel_batch3 = batches[16:24]

highLevel_batch4 = batches[24:32]
highLevel_batch5 = batches[32:40]
highLevel_batch6 = batches[40:48]

highLevel_batch7 = batches[48:56]
highLevel_batch8 = batches[56:64]
highLevel_batch9 = batches[64:72]

highLevel_batch10 = batches[72:80]
highLevel_batch11 = batches[80:88]
highLevel_batch12 = batches[88:96]

highLevel_batch13 = batches[96:104]
highLevel_batch14 = batches[104:112]
highLevel_batch15 = batches[112:]

########################################################################################################################
########################################################################################################################

# define which batch of spare parts will be used in this run
current_highLevelBatch = highLevel_batch1
current_highLevelBatch_name = 'highLevel_batch1'


# run loop through the batches
for b, batch in enumerate(current_highLevelBatch):
    print(f"Batch {b}")

    ## first we remove results from any previous iteration
    # Get a list of all variables in the current scope
    all_variables = list(locals().keys())

    # Iterate over each variable name
    for var_name in all_variables:
        if var_name.startswith('dict_') or var_name.startswith('df_out') or var_name.startswith('df_initial'):
            del locals()[var_name]

    gc.collect()

    # define which batch of spare parts will be used in this run
    currentBatch = current_highLevelBatch[b]
    currentBatch_name = f'batch{b}_parts'

    # load dictionary with demand data frames
    with open('Data//dict_demandPublished.pkl', 'rb') as f:
        dict_demand = pickle.load(f)

    # filter dict_demand for the spares we want to continue working with
    dict_demand = {k: v for k, v in dict_demand.items() if k in currentBatch}
    gc.collect()

    # create empty list to store information about the model configuration
    output = []

    # create empty list to store initial forecast of each part (last period of training data)
    list_initialForecast = []

    # create list to store spare parts for which the loop fails (gets error)
    list_failing_parts = []

    # create empty dictionary to store the forecasts in
    dict_forecast_RNN = {}


    # loop through spare parts
    for i, part in enumerate(currentBatch):
        print(f"Batch nr {b} van 7")
        print(part)
        try:
            print('\n current part is {}, current iteration is {}'.format(part, i))

            '''
            Tune the model
            '''
            # get part lead time and round up to whole week
            partLeadTime = math.ceil((df_price_leadtime[df_price_leadtime['part'] == part]['leadTime'].values[0]) / 7)
            # forecast horizon is partLeadTime plus one period (period is week, so plus one)
            forecastHorizon = partLeadTime + 1

            # set window size (number of periods that is looked back to determine patterns. Kind of arbitrary choice, just set to 52)
            window_size = 52

            # get demand data for this specific spare part and plit it into train and test set
            df_demand_part = dict_demand[part]
            df_train = df_demand_part[df_demand_part['date'] <= '2021-12-31']  # data prior to year 2022
            df_test = df_demand_part[df_demand_part['date'] > '2021-12-31']  # holdout sample: data of the year 2022

            # define start and end of test period (already here as we need them for the inital forecast right below)
            start_date = df_test['date'].iloc[0]
            end_date = df_test['date'].iloc[-1]

            '''
            To initialize the inventory system, we make an intial forecast at the end of week 51 in 2021. We assume an one week
            lead time for the initial order. So it will arrive at the end of week 52 and beginning of 2022 available to satisfy
            demand.
            For the initial forecast, the forecast horizon is L+R+1. +1 because the earliest moment a next order will be placed
            is the end of week 1 in 2022. so this initial order should also cover this first week of the test period.
            '''
            # get data before this week
            df_demandBeforeInitialForecast = df_demand_part[df_demand_part['date'] <= (start_date + datetime.timedelta(weeks=-2))]

            ## make model that can be used for the initial forecast
            # store demand data as numpy array
            initialForecast_data = df_demandBeforeInitialForecast['demand'].values

            # Prepare the data
            X_train, y_train = prepare_data(initialForecast_data, window_size=window_size, forecast_size=(forecastHorizon+1))

            # Define the model architecture (this could be improved by trial and error and evaluating the effect on the performance metrics)
            model = Sequential()
            model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(window_size, 1)))
            model.add(LSTM(1, activation='relu'))
            model.add(Dense((forecastHorizon+1)))

            # Compile the model
            model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            model.summary()

            # define model checkpoint criteria
            model_checkpoint = ModelCheckpoint('initialForecast_model_weights.h5', monitor='val_mse', save_best_only=True)

            # train the model
            history = model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train,
                                validation_split=0.2, epochs=100, verbose=2, callbacks=[model_checkpoint]) # put epochs lower than for model that is trained for the test period, as this model only is used once to make the initial forecast and this saves running time
            # save the lowest val_mse during the fitting process of the model
            min_valMSE = min(history.history['val_mse'])

            # Load best model weights
            best_model = load_model('initialForecast_model_weights.h5')


            # transform the data to a supervised learning problem. we don't have the y-variable yet, but only change the x-variable to this format (kept the y-variable out to ensure we don't use data we don't have at this point)
            X_Initialfcst = test_data_preperation(initialForecast_data, window_size=window_size)

            # reshape X_Initialfcst to have shape (batch_size, time_steps, num_features) and make predictions
            initialForecast = best_model.predict(X_Initialfcst.reshape((X_Initialfcst.shape[0], X_Initialfcst.shape[1], 1)))
            # we only save the last forecast, as this is the real initial forecast. All other forecasts are from the training period
            initialForecast = initialForecast[-1].tolist()

            # code to replace all negative value with 0 (as negative forecast is not possible in this empirical context)
            initialForecast = [num if num >= 0 else 0 for num in initialForecast]

            # Save results for the initial forecast
            list_initialForecast.append((part, initialForecast, min_valMSE))
            ''''
            IMPORTANT
            RNN forecasts directly over the whole period. It is not a level per period like Croston.
            Here we don't sum it already. we store the whole forecast as a list. In the inventory system, the sum of 
            this list should be taken. Note: don't multiply by forecastHorizon. As this model directly forecasts over the 
            whole horizon
            '''

            # store training and testing data as numpy array
            train_data = df_train['demand'].values

            # Prepare the data
            X_train, y_train = prepare_data(train_data, window_size=window_size, forecast_size=forecastHorizon)

            # Define the model
            model = Sequential()
            model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(window_size, 1)))
            model.add(LSTM(50, activation='relu'))
            model.add(Dense(forecastHorizon))

            # Compile the model
            model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            model.summary()

            # define model checkpoint criteria
            model_checkpoint = ModelCheckpoint('best_model_weights.h5', monitor='val_mse', save_best_only=True)


            # train the model
            history = model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train,
                      validation_split=0.2,epochs=200, verbose=2, callbacks=[model_checkpoint])
            # save the lowest val_mse during the fitting process of the model
            min_valMSE = min(history.history['val_mse'])

            # Load best model weights
            best_model = load_model('best_model_weights.h5')

            ''''
            Obtain the in-sample MAE of the naive forecast
            '''
            ## get MAE naive in sample
            start_dateTRAIN = df_train['date'].iloc[0]
            start_dateTRAIN = start_dateTRAIN + datetime.timedelta(weeks=(forecastHorizon - 1))  # we need the first week(s) for the first naive forecast. So we change start date to first date we can make the naive forecast (minus one because when we make the forecast we already have the demand data of the particular week)
            end_dateTRAIN = df_train['date'].iloc[-1]

            list_MAEs_naive = []
            # loop through each week of training data
            for dateMondayTRAIN in rrule.rrule(rrule.WEEKLY, dtstart=start_dateTRAIN, until=end_dateTRAIN):  # end_date
                # print(dateMondayTRAIN)

                # calculate the date of the last Monday in this forecast horizon (1 week + lead time)
                dateEndHorizon = dateMondayTRAIN + datetime.timedelta(weeks=forecastHorizon)

                # calculate the date of the Monday of the previous Monday (needed for calculating variance of demand)
                PreviousPeriodDateMonday = dateMondayTRAIN + datetime.timedelta(weeks=-1)

                if dateEndHorizon > end_dateTRAIN:  # if forecast goes further than 2021 (training data), we stop
                    continue
                else:
                    # get data up to and including this week
                    df_demandBeforePeriodTRAIN = df_train[df_train['date'] <= dateMondayTRAIN]
                    demandBeforePeriodTRAIN = df_demandBeforePeriodTRAIN['demand'].tolist()
                    naiveForecast = demandBeforePeriodTRAIN[-forecastHorizon:]

                    # get actual demand in that forecast horizon
                    naiveActuals = df_train.loc[(df_train['date'] > dateMondayTRAIN) & (df_train['date'] <= dateEndHorizon)]['demand'].tolist()

                    MAE_naive_horizonAheadPeriod = mae(y_true=(naiveActuals),
                                                       y_pred=naiveForecast)  # get the MAE for the naive forecast

                    list_MAEs_naive.append(MAE_naive_horizonAheadPeriod)

            # calculate the average of all MAEs for the naive forecast in the training set
            MAE_naive_horizonAhead = sum(list_MAEs_naive) / len(list_MAEs_naive)

            '''
            Loop through test period (2022)
            '''

            # get forecast using the best alpha value and store it in a dictionary
            # create columns in df_test to store information of the forecast
            df_test['forecast'] = np.nan
            df_test['forecast'] = df_test['forecast'] .astype('object') # change to datatype object, so we can save a list in it later on
            df_test['variance_forecast'] = np.nan
            df_test['periodMSE_horizonAhead'] = np.nan
            df_test['periodMASE_horizonAhead'] = np.nan

            # loop through each week of 2022 and forecast the demand for that week
            for dateMonday in rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date): #end_date
                #print(dateMonday)

                ## forecast future demand
                # calculate the date of the last Monday in this forecast horizon (1 week + lead time)
                dateEndHorizon = dateMonday + datetime.timedelta(weeks=forecastHorizon)

                # calculate the date of the Monday of the previous Monday (needed for calculating variance of demand)
                PreviousPeriodDateMonday = dateMonday + datetime.timedelta(weeks=-1)

                # if the end of the forecast horizon exceeds our test period (2022), we don't make new forecasts, but continue with our last forecast (if we don't do this, we cannot calculate the forecast accuracy metrics, since we don't have the actuals for 2023 (especially in case of very long lead times)
                if dateMonday.isocalendar().week == 52: # in the last week, we put NaN values for forecast and the forecast accuracy metrics. Because in the inventory system script for this week there will not be calculated a new order-up-to-level
                    #print('last week')
                    df_test.loc[df_test['date'] == dateMonday, 'forecast'] = np.nan
                    df_test.loc[df_test['date'] == dateMonday, 'periodMSE_horizonAhead'] = np.nan
                    df_test.loc[df_test['date'] == dateMonday, 'periodMASE_horizonAhead'] = np.nan
                elif dateEndHorizon > end_date:  # stop with make new forecast as the horizon of the forecast reaches the end of the test period (year 2022)
                    #print('forecast = lastForecast')
                    weeksLeft = 52 - dateMonday.isocalendar().week # get number of weeks left for which we need to use the old forecast
                    lastForecast = df_test.loc[df_test['date'] == (end_date - datetime.timedelta(weeks=forecastHorizon))]['forecast'].squeeze()
                    lastForecast = ast.literal_eval(lastForecast)

                    # keep from the last forecast only the week until and including week 51 (week 51 is forecast for week 52)
                    remainingForecast = lastForecast[-weeksLeft:]
                    #print('date of first week we do not make forecasts anymore', dateMonday)
                    df_test.loc[df_test['date'] == dateMonday, 'forecast'] = str(remainingForecast)
                    df_test.loc[df_test['date'] == dateMonday, 'periodMSE_horizonAhead'] = np.nan  # as we don't make new forecast, we don't calculate new forecast accuracy metrics
                    df_test.loc[df_test['date'] == dateMonday, 'periodMASE_horizonAhead'] = np.nan  # as we don't make new forecast, we don't calculate new forecast accuracy metrics

                else: # all other weeks we forecast as normale and calculate the forecast accuracy metrics
                    #print('normal week')
                    # get actual demand in that forecast horizon
                    demandActual =  df_test.loc[(df_test['date'] > dateMonday) & (df_test['date'] <= dateEndHorizon)]['demand'].tolist()
                    #print(demandActual)

                    # get data up to and including this week
                    df_demandBeforePeriod = df_demand_part[df_demand_part['date']<=dateMonday]

                    # store demand data as numpy array
                    testPeriodForecast_data = df_demandBeforePeriod['demand'].values

                    # transform the data to a supervised learning problem. we don't have the y-variable yet, but only change the x-variable to this format (kept the y-variable out to ensure we don't use data we don't have at this point)
                    X_forecast = test_data_preperation(testPeriodForecast_data, window_size=window_size)

                    # reshape X_Initialfcst to have shape (batch_size, time_steps, num_features) and make predictions
                    RNN_output = best_model.predict(X_forecast.reshape((X_forecast.shape[0], X_forecast.shape[1], 1)))
                    # we only save the last forecast/list, as this is the real initial forecast. All other forecasts are from the training period
                    forecast = RNN_output[-1].tolist()

                    # code to replace all negative value with 0 (as negative forecast is not possible in this empirical context)
                    forecast = [num if num >= 0 else 0 for num in forecast]

                    ''''
                    IMPORTANT
                    RNN forecasts directly over the whole period. It is not a level per period like Croston.
                    By summing it, we directly have the demand over the whole period. This forecast should in the 
                    inventory system NOT be multiplied by the forecastHorizon!!!
                    '''
                    # save forecast to dataframe, before we change it to np.array (because that will cause problems)
                    df_test.loc[df_test['date'] == dateMonday, 'forecast'] = str(forecast)

                    # change to numpy arrays and calculate MSE
                    demandActual = np.array(demandActual)
                    forecast = np.array(forecast)
                    periodMSE_horizonAhead = mean_squared_error(demandActual, forecast)

                    # calculate the MAE of the model's forecast
                    MAE_forecast_horizonAhead = mae(y_true=(demandActual.tolist()), y_pred=(forecast.tolist()))
                    # calculate the MASE for the forecast of this period
                    if MAE_naive_horizonAhead == 0:
                        periodMASE_horizonAhead = np.nan
                    else:
                        periodMASE_horizonAhead = MAE_forecast_horizonAhead / MAE_naive_horizonAhead  # divide the two MAEs to obtain the MASE (for further information see: https://medium.com/@ashishdce/mean-absolute-scaled-error-mase-in-forecasting-8f3aecc21968 or : Hyndman, R. & Koehler, A. (2006) Another look at measures of forecast accuracy. International Journal of Forecasting 22 (4), 679-688. https://doi.org/10.1016/j.ijforecast.2006.03.001)

                    # add to df_test the performance metrics
                    df_test.loc[df_test['date'] == dateMonday, 'periodMSE_horizonAhead'] = periodMSE_horizonAhead
                    df_test.loc[df_test['date'] == dateMonday, 'periodMASE_horizonAhead'] = periodMASE_horizonAhead

                ## calculate estimated variance of demand
                '''
                Below we forecast the variance of the demand as is done in the paper:
                
                Syntetos, A. A., Nikolopoulos, K., & Boylan, J. E. (2010). Judging the judges through accuracy-implication 
                metrics: The case of inventory forecasting. International Journal of Forecasting, 26(1), 134–143. 
                https://doi.org/10.1016/j.ijforecast.2009.05.016
                
                
                In the first weeks of our test period (2022) the problem exists that the actual and/or forecasted demand are 
                from 2021. In those cases we have to obtain some numbers in a different way
                '''

                # get date of t-n (for cumulative forecast)
                date_forecast_tMinusn = dateMonday - datetime.timedelta(weeks=forecastHorizon)
                # get date of t-1+1 (for summed actual demand over last n periods)
                date_actualDemand_i_tMinusNplusOne = dateMonday - datetime.timedelta(weeks=forecastHorizon) + datetime.timedelta(weeks=1)

                # in case we are in week 1: forecast = obtained from week 51 2021, actuals = only week 1 2022, PrevVariance = best_error_crossVal_MSE
                if dateMonday.isocalendar().week == 1:
                    # print('first if statement')
                    # get ACTUAL demand summed over period t-n+1 up to and including period t
                    actuals_i_tMinusNplusOne = df_test.loc[(df_test['date'] >= date_actualDemand_i_tMinusNplusOne) & (df_test['date'] <= dateMonday)]['demand'].tolist()
                    cumulative_i_tMinusNplusOne = sum(actuals_i_tMinusNplusOne)


                    # get cumulative FORECAST made in week 51 2021 in this case and multiply with number of actuals we have in this iteration
                    forecast_forecast_tMinusn = initialForecast
                    # calculate how many weeks of initial Forecast have become actuals
                    weeksLeft = dateMonday.isocalendar().week
                    # obtain only the part of the initial forecast that has become actual
                    remainingForecast = initialForecast[:weeksLeft]
                    cumulative_forecast_tMinusn = sum(remainingForecast)

                    # obtain the previous estimated variance of demand
                    PrevVariance = min_valMSE

                    # calculate the estimated variance of demand
                    variance_forecast = (0.25 * pow((cumulative_forecast_tMinusn - cumulative_i_tMinusNplusOne), 2)) + ((1 - 0.25) * PrevVariance)
                # in case only some weeks of t-n are in 2022
                elif date_actualDemand_i_tMinusNplusOne.year < 2022:
                    # print('second: first elif statement')
                    # get ACTUAL demand summed over period t-n+1 up to and including period t
                    actuals_i_tMinusNplusOne = df_test.loc[(df_test['date'] >= date_actualDemand_i_tMinusNplusOne) & (df_test['date'] <= dateMonday)]['demand'].tolist()
                    cumulative_i_tMinusNplusOne = sum(actuals_i_tMinusNplusOne)

                    # get cumulative FORECAST made in week 51 2021 in this case and multiply with number of actuals we have in this iteration
                    forecast_forecast_tMinusn = initialForecast
                    # calculate how many weeks of initial Forecast have become actuals
                    weeksLeft = dateMonday.isocalendar().week
                    # obtain only the part of the initial forecast that has become actual
                    remainingForecast = initialForecast[:weeksLeft]
                    cumulative_forecast_tMinusn = sum(remainingForecast)

                    # obtain the previous estimated variance of demand
                    PrevVariance = df_test.loc[df_test['date'] == PreviousPeriodDateMonday]['variance_forecast'].squeeze()  # get MSE of previous period

                    # calculate the estimated variance of demand
                    variance_forecast = (0.25 * pow((cumulative_forecast_tMinusn - cumulative_i_tMinusNplusOne), 2)) + ((1 - 0.25) * PrevVariance)
                # in case all weeks of actuals are in 2022 but the forecast is made in week 51 of 2021
                elif date_actualDemand_i_tMinusNplusOne.year == 2022 and date_forecast_tMinusn.year < 2022:
                    # print('third: second elif statement')
                    # get ACTUAL demand summed over period t-n+1 up to and including period t
                    actuals_i_tMinusNplusOne = df_test.loc[(df_test['date'] >= date_actualDemand_i_tMinusNplusOne) & (df_test['date'] <= dateMonday)]['demand'].tolist()
                    cumulative_i_tMinusNplusOne = sum(actuals_i_tMinusNplusOne)

                    # get cumulative FORECAST made in week 51 2021 in this case and multiply with number of actuals we have in this iteration
                    forecast_forecast_tMinusn = initialForecast
                    cumulative_forecast_tMinusn = sum(forecast_forecast_tMinusn)

                    # obtain the previous estimated variance of demand
                    PrevVariance = df_test.loc[df_test['date'] == PreviousPeriodDateMonday]['variance_forecast'].squeeze()  # get MSE of previous period

                    # calculate the estimated variance of demand
                    variance_forecast = (0.25 * pow((cumulative_forecast_tMinusn - cumulative_i_tMinusNplusOne), 2)) + ((1 - 0.25) * PrevVariance)
                # in case all weeks of actuals are in 2022 and the forecast is also made in 2022
                else:
                    # print('4th: else statement')
                    # get ACTUAL demand summed over period t-n+1 up to and including period t
                    actuals_i_tMinusNplusOne = df_test.loc[(df_test['date'] >= date_actualDemand_i_tMinusNplusOne) & (df_test['date'] <= dateMonday)]['demand'].tolist()
                    cumulative_i_tMinusNplusOne = sum(actuals_i_tMinusNplusOne)

                    # get cumulative FORECAST made at end period t-n-1 over the subsequent n periods
                    forecast_forecast_tMinusn = df_test.loc[df_test['date'] == date_forecast_tMinusn]['forecast'].squeeze()
                    forecast_forecast_tMinusn = ast.literal_eval(forecast_forecast_tMinusn)
                    cumulative_forecast_tMinusn = sum(forecast_forecast_tMinusn)

                    # obtain the previous estimated variance of demand
                    PrevVariance = df_test.loc[df_test['date'] == PreviousPeriodDateMonday]['variance_forecast'].squeeze()  # get MSE of previous period

                    # calculate the estimated variance of demand
                    variance_forecast = (0.25 * pow((cumulative_forecast_tMinusn - cumulative_i_tMinusNplusOne), 2)) + ((1 - 0.25) * PrevVariance)
                # print('over')

                # add to df_test the forecasted variance of the demand
                df_test.loc[df_test['date'] == dateMonday, 'variance_forecast'] = variance_forecast


            # average periodMSE_horizonAhead over whole test period to obtain the total MSE for the test period (average and not sum, as the number of forecasts differ between the different part since this depends on the lead time (longer lead time = higher forecastHorizon = less forecasts)
            total_periodMSE_horizonAhead = df_test['periodMSE_horizonAhead'].mean(skipna=True)
            total_periodMASE_horizonAhead = df_test['periodMASE_horizonAhead'].mean(skipna=True)

            ## MSE 1-period ahead
            # now we will calculate the periodMSE_oneAhead (so the error between the forecast and the actual demand of the subsequent period)
            list_forecastedDemand = [initialForecast[0]]
            # get the number of NaN values in column 'periodMSE_horizonAhead' (those are the number of weeks we didn't make a forecast, as the forecast horizon for these weeks exceeds 2022)
            num_nan = int(df_test['periodMSE_horizonAhead'].isna().sum())
            dateStoppedForecast = end_date - datetime.timedelta(weeks=num_nan)

            # loop through each week of 2022 until we stopped forecasting. of each week we will obtain the forecast for 1 period ahead and append it to list_forecastedDemand
            for dateMonday in rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=dateStoppedForecast):  # end_date
                #print(dateMonday)
                forecast_fullHorizon = df_test.loc[df_test['date'] == dateMonday]['forecast'].squeeze()
                forecast_fullHorizon = ast.literal_eval(forecast_fullHorizon)

                # obtain forecast for next period
                forecast_onePeriodAhead = forecast_fullHorizon[0]
                # append forecast one period ahead to list_forecastedDemand
                list_forecastedDemand.append(forecast_onePeriodAhead)

            list_actualDemand = df_test['demand'].tolist() # put demand data in list
            list_actualDemand = list_actualDemand[:(-int(num_nan)+1)] # remove last rows, but one less as forecast. Because the last forecast is for the period after. and the first demand period is from the initial forecast

            periodMSE_oneAhead = mean_squared_error(y_true=list_actualDemand,y_pred=list_forecastedDemand)

            ## MASE 1-period ahead
            # MASE calculations
            naiveForecast1periodAhead = df_train['demand'][:-1].tolist()  # get the naive forecast for the training period (forecast value of previous week)
            naiveActuals1periodAhead = df_train['demand'][1:].tolist()  # get the actuals for the naive forecast (could not be made for first period, so remove first period)
            MAE_naive_oneAhead = mae(y_true=(naiveActuals1periodAhead),
                                     y_pred=naiveForecast1periodAhead)  # get the MAE for the naive forecast

            MAE_forecast_oneAhead = mae(y_true=list_actualDemand,
                                        y_pred=list_forecastedDemand)  # get the MAE for the model's forecast
            if MAE_naive_oneAhead == 0:
                periodMASE_oneAhead = np.nan
            else:
                periodMASE_oneAhead = MAE_forecast_oneAhead / MAE_naive_oneAhead  # divide the two MAEs to obtain the MASE (for further information see: https://medium.com/@ashishdce/mean-absolute-scaled-error-mase-in-forecasting-8f3aecc21968 or : Hyndman, R. & Koehler, A. (2006) Another look at measures of forecast accuracy. International Journal of Forecasting 22 (4), 679-688. https://doi.org/10.1016/j.ijforecast.2006.03.001)

            # Count number of NaN for periodMASE_horizonAhead, so the number of times the MASE horizon ahead was not calculated (in case you want to anlayse this further later on)
            nan_count_MASE_horizonAhead = df_test['periodMASE_horizonAhead'].isna().sum()

            # save forecast in dictionary
            dict_forecast_RNN[part] = df_test

            # Save results about the test period
            output.append((part, min_valMSE, total_periodMSE_horizonAhead, periodMSE_oneAhead,
                           total_periodMASE_horizonAhead, nan_count_MASE_horizonAhead, periodMASE_oneAhead))

            gc.collect()

        except:
            output.append((part, None, None, None, None, None, None))
            list_failing_parts.append(part)


    #######################################################################################
    #                            4. Save results current batch                            #
    #######################################################################################
    # Convert output to dataframe
    df_output = pd.DataFrame(output, columns=['part', 'min_valMSE', 'total_periodMSE_horizonAhead','periodMSE_oneAhead',
                                              'total_periodMASE_horizonAhead','nan_count_MASE_horizonAhead','periodMASE_oneAhead'])

    # Save results
    df_output.to_csv('Output//model6B_RNN//{}//forecast_stats_RNN_{}.csv'.format(current_highLevelBatch_name, currentBatch_name), index=False)


    # Convert list_initialForecast to dataframe
    df_initialForecast = pd.DataFrame(list_initialForecast, columns=['part', 'forecast', 'min_valMSE'])

    # Save results
    df_initialForecast.to_csv('Output//model6B_RNN//{}//df_initialForecast_{}.csv'.format(current_highLevelBatch_name, currentBatch_name), index=False)

    # Save dictionary with predictions as pickle file
    with open('Output//model6B_RNN//{}//dict_forecast_RNN_{}.pkl'.format(current_highLevelBatch_name,currentBatch_name), 'wb') as f:
        pickle.dump(dict_forecast_RNN, f)



