# Imports
import pandas as pd
import numpy as np
import warnings
import pickle
from dateutil import rrule
import math
from scipy.stats import norm
import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
pd.options.mode.chained_assignment = None  # default='warn'

#######################################################################################
#                             1. Read and Preprocess Data                             #
#######################################################################################
# model 1_Croston
# load dictionary with forecasts
with open('Output//model1_Croston//dict_forecast_Croston.pkl', 'rb') as f:
    dict_demand_Croston = pickle.load(f)

# load data frame with lead time and price information
df_price_leadtime = pd.read_csv('Output//EDA//Demand//df_price_leadtime_part.csv')

# Save results
df_initialForecast = pd.read_csv('Output//model1_Croston//df_initialForecast_Croston.csv')

# put all spare parts that are forecasted in a list
parts = pd.read_csv('Data//df_all_parts.csv')
parts = parts[parts.columns[0]].values.tolist() # convert to list


#######################################################################################
#                              2. Implement inventory system                          #
#######################################################################################

# define the target service levels we want to loop through
list_targetServiceLevel =[.7, .8, .85, .9, .95, .99]

# create empty data frame to store information for all spare parts in
df_CSLcosts_ALLPARTS = pd.DataFrame(columns=['part', 'targetServiceLevel', 'achievedCSL', 'totalCosts'])

# create list to store spare parts for which the loop fails (gets error)
list_failing_parts = []

# create empty dictionary to store the forecasts in
dict_invenSystem_Croston = {}


# loop through spare parts
for i, part in enumerate(parts):

    try:
        print('\n current part is {}, current iteration is {}'.format(part, i))

        ## get all data needed for this specific part
        df_test = dict_demand_Croston[part]

        initialForecast_mean = df_initialForecast[df_initialForecast['part'] == part]['mean_forecast'].squeeze()
        initialForecast_variance = df_initialForecast[df_initialForecast['part'] == part]['best_error_crossVal_MSE'].squeeze()

        # get part lead time and round up to whole week
        partLeadTime = math.ceil((df_price_leadtime[df_price_leadtime['part'] == part]['leadTime'].values[0]) / 7)
        # get part price
        partPrice = df_price_leadtime[df_price_leadtime['part'] == part]['price'].squeeze()
        # in case the part price is a nan value in df_price_leadtime
        if np.isnan(partPrice):
            partPrice = df_price_leadtime['price'].mean()

        # first let's define the assumed costs
        fixedOrderCosts = 30  # per order
        inventoryHoldingCosts = (partPrice * 0.22) / 52  # per week per item (1 item per year is 22% of part's price. per week this should be divided by 52)
        shortageCost = 0  # per item

        # create empty data frame to store information for each target service level
        df_CSLcosts = pd.DataFrame(columns=['targetServiceLevel','achievedCSL','totalCosts','totalDemand','backorderCumulative','shortageCostsCumulative','orderCostsCumulative','holdingCostsCumulative'])

        for j, targetCSL in enumerate(list_targetServiceLevel):
            #print('\n current targetCSL is {}, current iteration is {}'.format(targetCSL, j))

            # define inventory figures we need to keep track off
            inventoryPosition = 0
            inventoryLevel = 0
            backorderCumulative = 0
            orderNR = 0

            # define cost figures we need to keep track off
            orderCostsCumulative = 0
            holdingCostsCumulative = 0
            shortageCostsCumulative = 0

            # create empty data frame to keep track off the orders placed
            df_orders = pd.DataFrame(columns=['orderNR','nrOfItems','orderDate','orderWeek','orderDeliveryDate'])

            # create empty data frame to keep track off changes in the inventory system
            df_invSystem = pd.DataFrame(columns=['dateMonday','itemsReceived','demand','backorders','backordersCarriedOver','backorderCumulative','OrderUpToLevel',
                                                 'orders','ENDinventoryPosition','ENDinventoryLevel',
                                                 'periodShortageCosts','shortageCostsCumulative',
                                                 'periodOrderCosts','orderCostsCumulative',
                                                 'periodHoldingCosts','holdingCostsCumulative'])

            # get start and end date of the test period (the year 2022)
            start_date = df_test['date'].iloc[0]
            end_date = df_test['date'].iloc[-1]

            # set some variables to 0 for df_invSystem
            backorders = 0
            periodShortageCosts = 0
            periodOrderCosts = 0

            '''
            to initialize the inventory system we first calculate an initial order-up-to-level 
            and set the inventory position and level of first period to this order-up-to-level 
            
            We multiply the initialForecast_mean by 2+partLeadTime, as the risk period consists of weeks that demand
            can be received until the next order arrives, so:
            - first week demand can be received, but first order is placed at end of this week 
            - lead time
            - next order will arrive a week later at the earliest
            '''
            ## initialize inventory system
            initialOrderUpToLevel = math.ceil((initialForecast_mean * (2+partLeadTime)) + (math.sqrt(initialForecast_variance * (2+partLeadTime)) * norm.ppf(targetCSL)))

            # calculate order costs for initial order
            periodOrderCosts = fixedOrderCosts + (partPrice * initialOrderUpToLevel)
            orderCostsCumulative = orderCostsCumulative + periodOrderCosts

            # loop through each week of 2022 and review the inventory policy
            for dateMonday in rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date): #end_date_test
                #print(dateMonday)

                # in first week, we have to consider the initalization of the system. Otherwise we can just continue with the most recent inventory figures
                if dateMonday.isocalendar().week == 1:
                    # event 1: receive orders
                    inventoryPosition = initialOrderUpToLevel
                    inventoryLevel = initialOrderUpToLevel


                    # event 2: observe demand and deduct this from IP and IL and count number of backorders if any. Also adjust the IP, IL and backorders for this demand
                    demand = df_test[df_test['date']==dateMonday]['demand'].squeeze()

                    if demand > inventoryLevel: # demand cannot be met
                        backorders = demand - max(0, inventoryLevel) # number of items backordered (when inventoryLevel < 0, we take zero, as we only want to consider the demand that couldn't be met from stock)
                        backordersCarriedOver = backorders # orders that could not be met are carried over to next period
                        backorderCumulative = backorderCumulative+backorders # update the total number of backorders

                        # calculate shortage costs
                        periodShortageCosts = backorders * shortageCost
                        shortageCostsCumulative = shortageCostsCumulative + periodShortageCosts # add shortage costs of this period to the total shortage costs
                    else:  # else demand is met and the backordersCarriedOver should be set to zero again (updating inventory levels is done in event 4)
                        backordersCarriedOver = 0

                    # event 3: get forecast of future demand and calculate new order-up-to-level
                    mean_forecast = df_test[df_test['date']==dateMonday]['mean_forecast'].squeeze()
                    variance_forecast = df_test[df_test['date'] == dateMonday]['variance_forecast'].squeeze()

                    # calculate new OrderUpToLevel based on forecast for coming period (lead time + review period). NOTE: no need to multiple variance with (L+R), as this is already estimated over the n subsequent periods (so the full forecast horizon). See Syntetos, A. A., Nikolopoulos, K., & Boylan, J. E. (2010) page 141
                    OrderUpToLevel = math.ceil((mean_forecast * (1 + partLeadTime)) + (math.sqrt(variance_forecast) * norm.ppf(targetCSL)))   #print(OrderUpToLevel)

                    # event 4: evaluate inventory position and level and place new order if necessary
                    inventoryPosition = inventoryPosition - demand # update inventoryPosition
                    inventoryLevel = inventoryLevel - demand # update inventoryLevel

                    orders = OrderUpToLevel - inventoryPosition # we order up to the order-up-to-level (this is the number of items to order)
                    if orders > 0: # if we have to place an order we take the next steps
                        # place order in df_orders
                        orderNR = orderNR+ 1 # counter of the order number
                        nrOfItems = orders
                        orderDate = dateMonday + datetime.timedelta(days=4) # get Friday of this week
                        orderWeek = dateMonday.isocalendar().week
                        orderDeliveryDate = orderDate + datetime.timedelta(weeks=partLeadTime)
                        df_orders = df_orders.append({'orderNR':orderNR,'nrOfItems':nrOfItems,'orderDate':orderDate,'orderWeek':orderWeek,'orderDeliveryDate':orderDeliveryDate}, ignore_index=True)

                        # update inventoryPosition
                        inventoryPosition = inventoryPosition + nrOfItems # add the outstanding orders to the inventoryPosition

                        # calculate order costs
                        periodOrderCosts = fixedOrderCosts + (partPrice * nrOfItems)
                        orderCostsCumulative = orderCostsCumulative + periodOrderCosts
                    elif orders <= 0: # set orders to zero if no order is placed (to get it correct in df_invSystem)
                        orders = 0

                    # calculate holding costs (only when inventoryLevel is positive, as otherwise no inventory was in stock)
                    if inventoryLevel > 0:
                        periodHoldingCosts = inventoryLevel * inventoryHoldingCosts
                        holdingCostsCumulative = holdingCostsCumulative + periodHoldingCosts

                    # append information about this iteration to df_invSystem
                    df_invSystem = df_invSystem.append(
                        {'dateMonday':dateMonday,'itemsReceived':np.nan,'demand':demand,'OrderUpToLevel':OrderUpToLevel,
                         'ENDinventoryPosition':inventoryPosition,'ENDinventoryLevel':inventoryLevel,
                         'orders':orders,'backorders':backorders,'backorderCumulative':backorderCumulative,
                         'periodShortageCosts':periodShortageCosts,'shortageCostsCumulative':shortageCostsCumulative,
                         'periodOrderCosts':periodOrderCosts,'orderCostsCumulative':orderCostsCumulative,
                         'periodHoldingCosts':periodHoldingCosts,'holdingCostsCumulative':holdingCostsCumulative,
                         'backordersCarriedOver':backordersCarriedOver}, ignore_index=True)


                    # in the last week, things go a little differently because, among other things, we don't have to calculate a new order-up-to-leveles
                elif dateMonday.isocalendar().week == 52:
                    # event 1: receive orders
                    itemsReceived = df_orders[df_orders['orderDeliveryDate']<= dateMonday]['nrOfItems'].sum() # determine how many orders are received (last Friday, so let's say at start of this week)
                    df_orders = df_orders[df_orders['orderDeliveryDate'] > dateMonday] # remove orders received from df_orders by filtering out the orders that are left

                    inventoryLevel = inventoryLevel + itemsReceived # update inventoryLevel (inventoryPosition is already raised when the order was placed)


                    # event 2: observe demand and deduct this from IP and IL and count number of backorders if any. Also adjust the IP, IL and backorders for this demand
                    # in case of positive inventoryLevel, reduce backordersCarriedOver as much as possible
                    if inventoryLevel > 0 :
                        if inventoryLevel >= backordersCarriedOver: # all backordersCarriedOver can be fulfilled
                            backordersCarriedOver = 0 # all backordersCarriedOver are fulfilled, so set to zero
                            inventoryLevel = inventoryLevel-backordersCarriedOver # calculate what is left over from inventoryLevel
                        elif inventoryLevel < backordersCarriedOver: # not all backordersCarriedOver can be fulfilled
                            backordersCarriedOver = backordersCarriedOver - inventoryLevel # calculate how many backordersCarriedOver are left
                            inventoryLevel = 0 # all inventoryLevel is used, so set to zero

                    demand = df_test[df_test['date'] == dateMonday]['demand'].squeeze()

                    if demand > inventoryLevel:  # demand cannot be met
                        backorders = demand -  max(0, inventoryLevel) # number of items backordered (when inventoryLevel < 0, we take zero, as we only want to consider the demand that couldn't be met from stock)
                        backordersCarriedOver = backordersCarriedOver + backorders # orders that could not be met are carried over to next period
                        backorderCumulative = backorderCumulative + backorders  # update the total number of backorders

                        # calculate shortage costs
                        periodShortageCosts = backorders * shortageCost
                        shortageCostsCumulative = shortageCostsCumulative + periodShortageCosts  # add shortage costs of this period to the total shortage costs

                    # event 3: get forecast of future demand and calculate new order-up-to-level
                    ## as this is the last week, we skip this step

                    # event 4: evaluate inventory position and level and place new order if necessary
                    inventoryPosition = inventoryPosition - demand  # update inventoryPosition
                    inventoryLevel = inventoryLevel - demand  # update inventoryLevel

                    # since this is the last week, we do not order new items


                    # calculate holding costs (only when inventoryLevel is positive, as otherwise no inventory was in stock)
                    if inventoryLevel > 0:
                        periodHoldingCosts = inventoryLevel * inventoryHoldingCosts
                        holdingCostsCumulative = holdingCostsCumulative + periodHoldingCosts

                    # append information about this iteration to df_invSystem
                    df_invSystem = df_invSystem.append(
                        {'dateMonday':dateMonday,'itemsReceived':itemsReceived,'demand':demand,'OrderUpToLevel':OrderUpToLevel,
                         'ENDinventoryPosition':inventoryPosition,'ENDinventoryLevel':inventoryLevel,
                         'orders':np.nan,'backorders':backorders,'backorderCumulative':backorderCumulative,
                         'periodShortageCosts':periodShortageCosts,'shortageCostsCumulative':shortageCostsCumulative,
                         'periodOrderCosts':np.nan,'orderCostsCumulative':orderCostsCumulative,
                         'periodHoldingCosts':periodHoldingCosts,'holdingCostsCumulative':holdingCostsCumulative,
                         'backordersCarriedOver': backordersCarriedOver}, ignore_index=True)

                # below is for all others week than the first or last week
                else:
                    # event 1: receive orders
                    itemsReceived = df_orders[df_orders['orderDeliveryDate']<= dateMonday]['nrOfItems'].sum() # determine how many orders are received (last Friday, so let's say at start of this week)
                    df_orders = df_orders[df_orders['orderDeliveryDate'] > dateMonday] # remove orders received from df_orders by filtering out the orders that are left

                    inventoryLevel = inventoryLevel + itemsReceived # update inventoryLevel (inventoryPosition is already raised when the order was placed)

                    # event 2: observe demand and deduct this from IP and IL and count number of backorders if any. Also adjust the IP, IL and backorders for this demand
                    # in case of positive inventoryLevel, reduce backordersCarriedOver as much as possible
                    if inventoryLevel > 0 :
                        if inventoryLevel >= backordersCarriedOver: # all backordersCarriedOver can be fulfilled
                            backordersCarriedOver = 0 # all backordersCarriedOver are fulfilled, so set to zero
                            inventoryLevel = inventoryLevel-backordersCarriedOver # calculate what is left over from inventoryLevel
                        elif inventoryLevel < backordersCarriedOver: # not all backordersCarriedOver can be fulfilled
                            backordersCarriedOver = backordersCarriedOver - inventoryLevel # calculate how many backordersCarriedOver are left
                            inventoryLevel = 0 # all inventoryLevel is used, so set to zero

                    # demand during this period
                    demand = df_test[df_test['date'] == dateMonday]['demand'].squeeze()

                    if demand > inventoryLevel:  # demand cannot be met
                        backorders = demand - max(0, inventoryLevel) # number of items backordered (when inventoryLevel < 0, we take zero, as we only want to consider the demand that couldn't be met from stock)
                        backordersCarriedOver = backordersCarriedOver + backorders # orders that could not be met are carried over to next period
                        backorderCumulative = backorderCumulative + backorders # update the total number of backorders

                        # calculate shortage costs
                        periodShortageCosts = backorders * shortageCost
                        shortageCostsCumulative = shortageCostsCumulative + periodShortageCosts  # add shortage costs of this period to the total shortage costs
                        #(updating inventory levels is done in event 4)


                    # event 3: get forecast of future demand and calculate new order-up-to-level
                    mean_forecast = df_test[df_test['date']==dateMonday]['mean_forecast'].squeeze()
                    variance_forecast = df_test[df_test['date'] == dateMonday]['variance_forecast'].squeeze()

                    # calculate new OrderUpToLevel based on forecast for coming period (lead time + review period). NOTE: no need to multiple variance with (L+R), as this is already estimated over the n subsequent periods (so the full forecast horizon). See Syntetos, A. A., Nikolopoulos, K., & Boylan, J. E. (2010) page 141
                    OrderUpToLevel = math.ceil((mean_forecast * (1 + partLeadTime)) + (math.sqrt(variance_forecast) * norm.ppf(targetCSL)))
                    # print(OrderUpToLevel)


                    # event 4: evaluate inventory position and level and place new order if necessary
                    inventoryPosition = inventoryPosition - demand # update inventoryPosition
                    inventoryLevel = inventoryLevel - demand # update inventoryLevel

                    orders = OrderUpToLevel - inventoryPosition # we order up to the order-up-to-level (this is the number of items to order)
                    if orders > 0: # if we have to place an order we take the next steps
                        # place order in df_orders
                        orderNR = orderNR+ 1 # counter of the order number
                        nrOfItems = orders
                        orderDate = dateMonday + datetime.timedelta(days=4) # get Friday of this week
                        orderWeek = dateMonday.isocalendar().week
                        orderDeliveryDate = orderDate + datetime.timedelta(weeks=partLeadTime)
                        df_orders = df_orders.append({'orderNR':orderNR,'nrOfItems':nrOfItems,'orderDate':orderDate,'orderWeek':orderWeek,'orderDeliveryDate':orderDeliveryDate}, ignore_index=True)

                        # update inventoryPosition
                        inventoryPosition = inventoryPosition + nrOfItems # add the outstanding orders to the inventoryPosition

                        # calculate order costs
                        periodOrderCosts = fixedOrderCosts + (partPrice * nrOfItems)
                        orderCostsCumulative = orderCostsCumulative + periodOrderCosts
                    elif orders <= 0: # set orders to zero if no order is placed (to get it correct in df_invSystem)
                        orders = 0

                    # calculate holding costs (only when inventoryLevel is positive, as otherwise no inventory was in stock)
                    if inventoryLevel > 0:
                        periodHoldingCosts = inventoryLevel * inventoryHoldingCosts
                        holdingCostsCumulative = holdingCostsCumulative + periodHoldingCosts

                    # append information about this iteration to df_invSystem
                    df_invSystem = df_invSystem.append(
                        {'dateMonday':dateMonday,'itemsReceived':itemsReceived,'demand':demand,'OrderUpToLevel':OrderUpToLevel,
                         'ENDinventoryPosition':inventoryPosition,'ENDinventoryLevel':inventoryLevel,
                         'orders':orders,'backorders':backorders,'backorderCumulative':backorderCumulative,
                         'periodShortageCosts':periodShortageCosts,'shortageCostsCumulative':shortageCostsCumulative,
                         'periodOrderCosts':periodOrderCosts,'orderCostsCumulative':orderCostsCumulative,
                         'periodHoldingCosts':periodHoldingCosts,'holdingCostsCumulative':holdingCostsCumulative,
                         'backordersCarriedOver':backordersCarriedOver}, ignore_index=True)


            # calculate the totals for this target service level
            totalDemand = df_test['demand'].sum()
            if totalDemand > 0:
                achievedCSL = (totalDemand - backorderCumulative) / totalDemand
            else: # there was no demand during the test period, in this case we set the achievedCSL to 100%. But we still have to look what costs go along with this
                achievedCSL = 1.0

            # calculate the total costs by adding up all different costs
            totalCosts = shortageCostsCumulative + orderCostsCumulative + holdingCostsCumulative

            # store information for this target service level in df_CSLcosts
            df_CSLcosts = df_CSLcosts.append({'targetServiceLevel':targetCSL,'achievedCSL':achievedCSL,'totalCosts':totalCosts,
                                              'totalDemand':totalDemand,'backorderCumulative':backorderCumulative,
                                              'shortageCostsCumulative':shortageCostsCumulative,'orderCostsCumulative':orderCostsCumulative,
                                              'holdingCostsCumulative':holdingCostsCumulative}, ignore_index=True)

            # store information for this target service level in df_CSLcosts_ALLPARTS
            df_CSLcosts_ALLPARTS = df_CSLcosts_ALLPARTS.append(
                {'part': part, 'targetServiceLevel': targetCSL, 'achievedCSL': achievedCSL, 'totalCosts': totalCosts},
                ignore_index=True)

        # save info about inventory system in dictionary
        dict_invenSystem_Croston[part] = df_CSLcosts

    except:
        df_CSLcosts_ALLPARTS = df_CSLcosts_ALLPARTS.append({'part': part, 'targetServiceLevel': np.nan, 'achievedCSL': np.nan, 'totalCosts': np.nan},ignore_index=True)
        list_failing_parts.append(part)




#######################################################################################
#                          3. Save results inventory system                           #
#######################################################################################
# Save results
df_CSLcosts_ALLPARTS.to_csv('Output//model1_Croston//invSystemdf_CSLcosts_ALLPARTS_.csv', index=False)

# Save dictionary with predictions as pickle file
with open('Output//model1_Croston//dict_invSystemdf_dict_invenSystem_Croston.pkl', 'wb') as f:
    pickle.dump(dict_invenSystem_Croston, f)

