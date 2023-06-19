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
#                 2. Define function for the Croston's method                         #
#######################################################################################

# define function. source: https://towardsdatascience.com/croston-forecast-model-for-intermittent-demand-360287a17f5f
def Croston(ts, alpha, extra_periods):
    d = np.array(ts)  # Transform the input into a numpy array

    cols = len(d)  # Historical period length
    d = np.append(d, [np.nan] * extra_periods)  # Append np.nan into the demand array to cover future periods

    # level (a), periodicity(p) and forecast (f)
    a, p, f = np.full((3, cols + extra_periods), np.nan)
    q = 1  # periods since last demand observation

    # Initialization
    first_occurence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurence]
    p[0] = 1 + first_occurence
    f[0] = a[0] / p[0]
    # Create all the t+1 forecasts
    for t in range(0, cols):
        if d[t] > 0:
            a[t + 1] = alpha * d[t] + (1 - alpha) * a[t] # level     fc.y
            p[t + 1] = alpha * q + (1 - alpha) * p[t] # period  fc.tau
            f[t + 1] = a[t + 1] / p[t + 1]
            q = 1
        else:
            a[t + 1] = a[t]
            p[t + 1] = p[t]
            f[t + 1] = f[t]
            q += 1

    # Future Forecast
    a[cols + 1:cols + extra_periods] = a[cols]
    p[cols + 1:cols + extra_periods] = p[cols]
    f[cols + 1:cols + extra_periods] = f[cols]

    df = pd.DataFrame.from_dict({"Demand": d, "Forecast": f, "Period": p, "Level": a, "Error": d - f})
    return df

#######################################################################################
#         3. Define function for hyperparameter tuning using cross-validation         #
#######################################################################################
def rolling_forecast_origin(train, min_train_size, horizon):
    '''
    Rolling forecast origin generator
    :param train: training data as series
    :param min_trainsize: integer specifying minimal length of expected training set
    :param horizon:  integer specifying forecast horizion
    :return: split_train, split_val: 2 series
    '''
    for i in range(len(train) - min_train_size - horizon + 1):
        split_train = train[:min_train_size+i]
        split_val = train[min_train_size+i:min_train_size+i+horizon]
        yield split_train, split_val

# function for finding average of list
def Average(lst):
    # average function
    avg = np.average(lst)
    return (avg)


def cross_validation_score_croston(train, cv,alpha_value):  #JE HEBT IETS ANDERS T.O.V. VIDEO. BV METRIC NIET (VIDEO: https://www.youtube.com/watch?v=g9iO2AwTXyI&ab_channel=HSMA)
    '''
    Calculate cross validation scores
    '''
    cv_scores = []
    for cv_train, cv_val in cv:
        #print(cv_val) DIT KAN WEG
        preds = Croston(cv_train, alpha=alpha_value, extra_periods=len(cv_val))
        forecast = preds['Forecast'].iloc[-len(cv_val):].to_numpy()
        meanSquaredErrorPrediction = mean_squared_error(cv_val, forecast)
        #squaredError =  pow((cv_val[0]-forecast),2) DIT KAN WEG
        cv_scores.append(meanSquaredErrorPrediction)
    meanSquaredError = Average(cv_scores) # calculate the average MSE. average as the length of the training set differ between parts. when summing, this would mean that differences in the total are also caused by different number of MSE's to sum
    return meanSquaredError


#######################################################################################
#                      4. Apply DLP model to the spare parts                          #
#######################################################################################
# create list for alpha values that will be tried for Croston's method
alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# create empty list to store information about the model configuration
output = []

# create empty list to store initial forecast of each part (last period of training data)
list_initialForecast = []

# create list to store spare parts for which the loop fails (gets error)
list_failing_parts = []

# create empty dictionary to store the forecasts in
dict_forecast_DLP = {}

# loop through spare parts
for i, part in enumerate(parts):

    try:
        print('\n current part is {}, current iteration is {}'.format(part, i))

        '''
        Tune the model
        '''
        # Create a list to store model result for each alpha value
        order_error = []

        # get part lead time and round up to whole week
        partLeadTime =  math.ceil((df_price_leadtime[df_price_leadtime['part']==part]['leadTime'].values[0])/7)
        # forecast horizon is partLeadTime plus one period (period is week, so plus one)
        forecastHorizon = partLeadTime + 1

        # get demand data for this specific spare part and plit it into train and test set
        df_demand_part = dict_demand[part]
        df_train = df_demand_part[df_demand_part['date'] <= '2021-12-31']  # data prior to year 2022
        df_test = df_demand_part[df_demand_part['date'] > '2021-12-31']  # holdout sample: data of the year 2022

        # fit model in a loop and try different alpha values
        for j, alpha in enumerate(alpha_values):
            # print('\n current alpha value is {}, current iteration is {}'.format(alpha, j))

            # store training data as numpy array
            train = df_train['demand'].values

            # create rolling forecast
            cv_rolling = rolling_forecast_origin(train=train, min_train_size=2, horizon=forecastHorizon)
            # run cross validation and get the MSE
            cv_scores_Croston = cross_validation_score_croston(train=train, cv=cv_rolling, alpha_value=alpha)

            # append tuning value and corresponding performance to list
            order_error.append((alpha, cv_scores_Croston))

        df_order_error = pd.DataFrame(order_error,columns=['alpha', 'crossVal_MSE'])

        # Best parameters based on crossVal_MSE
        best_alpha = df_order_error.sort_values('crossVal_MSE').head(1)[['alpha']].squeeze()
        best_error_crossVal_MSE = df_order_error.sort_values('crossVal_MSE').head(1)[['crossVal_MSE']].squeeze()

        ''''
        Obtain the in-sample MAE of the naive forecast
        '''
        ## get MAE naive in sample
        start_dateTRAIN = df_train['date'].iloc[0]
        start_dateTRAIN = start_dateTRAIN + datetime.timedelta(weeks=(forecastHorizon - 1))  # we need first weeks for first naive forecast, so we change start date to first date we can make the naive forecast (minus one because when we make the forecast we already have the demand data of the particular week)
        end_dateTRAIN = df_train['date'].iloc[-1]

        list_MAEs_naive = []
        # loop through each week of training data
        for dateMondayTRAIN in rrule.rrule(rrule.WEEKLY, dtstart=start_dateTRAIN, until=end_dateTRAIN):  # end_date
            # print(dateMondayTRAIN)

            # calculate the date of the last Monday in this forecast horizon (1 week + lead time)
            dateEndHorizon = dateMondayTRAIN + datetime.timedelta(weeks=forecastHorizon)

            # calculate the date of the Monday of the previous Monday (needed for calculating variance of demand)
            PreviousPeriodDateMonday = dateMondayTRAIN + datetime.timedelta(weeks=-1)

            if dateEndHorizon > end_dateTRAIN:  # if forecast goes further than 2021, we stop
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
        # get first and last Monday of test data/period
        start_date = df_test['date'].iloc[0]
        end_date = df_test['date'].iloc[-1]

        # create columns in df_test to store information of the forecast
        df_test['forecast'] = np.nan
        df_test['variance_forecast'] = np.nan
        df_test['periodMSE_horizonAhead'] = np.nan
        df_test['periodMASE_horizonAhead'] = np.nan

        '''
        To initialize the inventory system, we make an intial forecast at the end of week 51 in 2021. We assume an one week
        lead time for the initial order. So it will arrive at the end of week 52 and beginning of 2022 available to satisfy
        demand.
        For the initial forecast, the forecast horizon is L+R+1. +1 because the earliest moment a next order will be placed
        is the end of week 1 in 2022. so this initial order should also cover this first week of the test period.
        '''
        # get data before this week
        df_demandBeforeInitialForecast = df_demand_part[df_demand_part['date'] <= (start_date + datetime.timedelta(weeks=-2))]

        # 1. make forecast using Croston
        croston_output = Croston(ts=df_demandBeforeInitialForecast['demand'], alpha=best_alpha, extra_periods=(forecastHorizon+1))
        # 2. get variables needed
        period = croston_output.Period[-1:].squeeze()
        level = croston_output.Level[-1:].squeeze()

        df_demandBeforePeriod = df_demandBeforeInitialForecast.set_index('date')  # set the column 'date' as index
        non_zero_indices = np.nonzero((df_demandBeforePeriod.demand.to_numpy()))[0]  # Find the non-zero indices which can be used to calculate the number of periods since last demand observation
        sinceLast = len(df_demandBeforePeriod) - non_zero_indices[-1] - 1  # calculate number of periods since last demand occurance (-1 is correct, otherwise we end up with value that is 1 to high)

        # 3. make forecast using DLP (using Croston as input)
        initialForecast = (level / period) * ((forecastHorizon+1) + (sinceLast - (1 - (1 / period)) / (1 / period)) * (1 - (1 - (1 / period)) ** (forecastHorizon+1)))

        # Save results for the initial forecast
        list_initialForecast.append((part, initialForecast, best_error_crossVal_MSE))


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
                lastForecast = df_test.loc[df_test['date'] == (end_date - datetime.timedelta(weeks=forecastHorizon))]['forecast'].squeeze() # get last forecast (which is over the full period)
                lastForecastPeriodically = lastForecast / forecastHorizon # divide the last forecast by the forecastHorizon, to convert this from a forecast of over the full forecastHorizon to a periodic (weekly) forecast

                # keep from the last forecast only the week until and including week 51 (week 51 is forecast for week 52)
                remainingForecast = lastForecastPeriodically * weeksLeft

                #print('date of first week we do not make forecasts anymore', dateMonday)
                df_test.loc[df_test['date'] == dateMonday, 'forecast'] = remainingForecast
                df_test.loc[df_test['date'] == dateMonday, 'periodMSE_horizonAhead'] = np.nan  # as we don't make new forecast, we don't calculate new forecast accuracy metrics
                df_test.loc[df_test['date'] == dateMonday, 'periodMASE_horizonAhead'] = np.nan  # as we don't make new forecast, we don't calculate new forecast accuracy metrics

            else: # all other weeks we forecast as normale and calculate the forecast accuracy metrics
                #print('normal week')
                # get actual demand in that forecast horizon
                demandActual =  df_test.loc[(df_test['date'] > dateMonday) & (df_test['date'] <= dateEndHorizon)]['demand'].tolist()
                #print(demandActual)

                # get data up to and including this week
                df_demandBeforePeriod = df_demand_part[df_demand_part['date']<=dateMonday]

                # forecast demand at start of period for coming forecast horizon
                # 1. make forecast using Croston
                croston_output = Croston(ts=df_demandBeforePeriod['demand'], alpha=best_alpha,extra_periods=forecastHorizon)
                # 2. get variables needed
                period = croston_output.Period[-1:].squeeze()
                level = croston_output.Level[-1:].squeeze()

                df_demandBeforePeriod = df_demandBeforePeriod.set_index('date') # set the column 'date' as index
                non_zero_indices = np.nonzero((df_demandBeforePeriod.demand.to_numpy()))[0] # Find the non-zero indices which can be used to calculate the number of periods since last demand observation
                sinceLast = len(df_demandBeforePeriod) - non_zero_indices[-1] - 1 # calculate number of periods since last demand occurance (-1 is correct, otherwise we end up with value that is 1 to high)

                # 3. make forecast using DLP (using Croston as input)
                forecast = (level/period) * (forecastHorizon + (sinceLast - (1-(1/period)) / (1/period)) * (1-  (1-(1/period))**forecastHorizon))
                ''''
                IMPORTANT
                DLP forecasts directly over the whole period. It is not a level per period like Croston.
                This forecast should in the inventory system NOT be multiplied by the forecastHorizon!
                '''

                # change to numpy arrays and calculate MSE
                demandActual = np.array(demandActual)
                list_forecast = [forecast/forecastHorizon] * forecastHorizon # get DLP forecast per period (average weekly forecast over the next L+R weeks)
                periodicForecast = np.array(list_forecast)
                periodMSE_horizonAhead = mean_squared_error(y_true=demandActual, y_pred=periodicForecast)

                # calculate the MAE of the model's forecast
                MAE_forecast_horizonAhead = mae(y_true=(demandActual.tolist()), y_pred=(list_forecast))
                # calculate the MASE for the forecast of this period
                if MAE_naive_horizonAhead == 0:
                    periodMASE_horizonAhead = np.nan
                else:
                    periodMASE_horizonAhead = MAE_forecast_horizonAhead / MAE_naive_horizonAhead  # divide the two MAEs to obtain the MASE (for further information see: https://medium.com/@ashishdce/mean-absolute-scaled-error-mase-in-forecasting-8f3aecc21968 or : Hyndman, R. & Koehler, A. (2006) Another look at measures of forecast accuracy. International Journal of Forecasting 22 (4), 679-688. https://doi.org/10.1016/j.ijforecast.2006.03.001)


                # add to df_test the forecast and performance metrics
                df_test.loc[df_test['date'] == dateMonday, 'forecast'] = forecast
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
                initialForecastPeriodically = initialForecast / (forecastHorizon+1)  # divide the last forecast by (forecastHorizon+1), to convert this from a forecast of over the full forecastHorizon + 1 to a periodic (weekly) forecast
                # calculate how many weeks of initial Forecast have become actuals
                weeksLeft = dateMonday.isocalendar().week
                # obtain only the part of the initial forecast that has become actual
                remainingForecast = initialForecastPeriodically*weeksLeft
                cumulative_forecast_tMinusn = remainingForecast

                # obtain the previous estimated variance of demand
                PrevVariance = best_error_crossVal_MSE

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
                initialForecastPeriodically = initialForecast / (forecastHorizon+1)  # divide the last forecast by (forecastHorizon+1), to convert this from a forecast of over the full forecastHorizon + 1 to a periodic (weekly) forecast
                # calculate how many weeks of initial Forecast have become actuals
                weeksLeft = dateMonday.isocalendar().week
                # obtain only the part of the initial forecast that has become actual
                remainingForecast = initialForecastPeriodically * weeksLeft
                cumulative_forecast_tMinusn = remainingForecast

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
                initialForecastPeriodically = initialForecast / (forecastHorizon + 1)  # divide the last forecast by (forecastHorizon+1), to convert this from a forecast of over the full forecastHorizon + 1 to a periodic (weekly) forecast
                # calculate how many weeks of initial Forecast have become actuals
                weeksLeft = dateMonday.isocalendar().week
                # obtain only the part of the initial forecast that has become actual
                remainingForecast = initialForecastPeriodically * weeksLeft
                cumulative_forecast_tMinusn = remainingForecast

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
                cumulative_forecast_tMinusn = forecast_forecast_tMinusn

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
        list_forecastedDemand = [(initialForecast/(forecastHorizon + 1))]  # initiate list with forecast for first week

        # get the number of NaN values in column 'periodMSE_horizonAhead' (those are the number of weeks we didn't make a forecast, as the forecast horizon for these weeks exceeds 2022)
        num_nan = float(df_test['periodMSE_horizonAhead'].isna().sum())
        dateStoppedForecast = end_date - datetime.timedelta(weeks=num_nan)

        # loop through each week of 2022 until we stopped forecasting. of each week we will obtain the forecast for 1 period ahead and append it to list_forecastedDemand
        for dateMonday in rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=dateStoppedForecast):  # end_date
            #print(dateMonday)
            forecast_fullHorizon = df_test.loc[df_test['date'] == dateMonday]['forecast'].squeeze()

            # obtain forecast for next period
            forecast_onePeriodAhead = forecast_fullHorizon/forecastHorizon
            # append forecast one period ahead to list_forecastedDemand
            list_forecastedDemand.append(forecast_onePeriodAhead)

        list_actualDemand = df_test['demand'].tolist()  # put demand data in list
        list_actualDemand = list_actualDemand[:(-int(num_nan) + 1)]  # remove last rows, but one less as forecast. Because the last forecast is for the period after. and the first demand period is from the initial forecast

        periodMSE_oneAhead = mean_squared_error(y_true=list_actualDemand, y_pred=list_forecastedDemand)

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
        dict_forecast_DLP[part] = df_test

        # Save results about the test period
        output.append((part, best_alpha, best_error_crossVal_MSE, total_periodMSE_horizonAhead, periodMSE_oneAhead,
                       total_periodMASE_horizonAhead, nan_count_MASE_horizonAhead, periodMASE_oneAhead))

    except:
        output.append((part, None, None, None, None, None, None, None))
        list_failing_parts.append(part)



#######################################################################################
#                             5. Save results DLP method                              #
#######################################################################################
# Convert output to dataframe
df_output = pd.DataFrame(output, columns=['part', 'best_alpha', 'best_error_crossVal_MSE', 'total_periodMSE_horizonAhead','periodMSE_oneAhead',
                                          'total_periodMASE_horizonAhead','nan_count_MASE_horizonAhead','periodMASE_oneAhead'])

# Remove rows with NaN values in 'Name' and 'Salary' columns
df_output = df_output.dropna(subset=['best_alpha', 'best_error_crossVal_MSE'])

# Save results
df_output.to_csv('Output//model5_DLP//forecast_stats_DLP.csv', index=False)


# Convert list_initialForecast to dataframe
df_initialForecast = pd.DataFrame(list_initialForecast, columns=['part', 'forecast','best_error_crossVal_MSE'])

# Save results
df_initialForecast.to_csv('Output//model5_DLP//df_initialForecast_DLP.csv', index=False)


# Save dictionary with predictions as pickle file
with open('Output//model5_DLP//dict_forecast_DLP.pkl', 'wb') as f:
    pickle.dump(dict_forecast_DLP, f)

