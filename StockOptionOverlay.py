import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm


SOFR_history = pd.read_csv(r'C:\Users\rensi\Desktop\Siddhartha Folder\Fordham University\Semester 1\QFGB 8946 Financial Markets and Modeling\Homework\HW8\SOFR_history.csv')

SOFR_history['Date'] = pd.to_datetime(SOFR_history['Effective Date'], format='%m/%d/%Y')
SOFR_history.set_index('Date', inplace=True)

SOFR_history = SOFR_history.resample('MS').first()

SOFR_history = SOFR_history['Rate (%)']

SOFR_history = SOFR_history.to_frame().rename(columns={'Rate (%)': 'Rate'})

SOFR_history

start_date = "2022-11-01"
end_date = "2023-12-01"

df_vix = yf.download('^VIX', start = start_date, end = end_date)
df_vix = df_vix.resample('MS').first()  

df_vix = df_vix['Adj Close']

df_vix = df_vix.to_frame().rename(columns={'Rate (%)': 'Rate'})
df_vix.rename(columns={'Adj Close': 'Adj Close_Vix'}, inplace=True)




df_spy = yf.download('SPY',start = start_date, end = end_date)
df_spy = df_spy.resample('MS').first()  

df_spy = df_spy['Adj Close']
df_spy = df_spy.to_frame().rename(columns={'Rate (%)': 'Rate'})
df_spy.rename(columns={'Adj Close': 'Adj Close_Spy'}, inplace=True)


def call_black_scholes(S, K, sigma, r, T, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)  
    
    call_price = S * np.exp(-r * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price

def put_black_scholes(S, K, sigma, r, T, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)  
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-r * T) * norm.cdf(-d1)
    
    return put_price


data = SOFR_history.join([df_vix, df_spy], how='inner')

data['K_C'] = data['Adj Close_Spy']+data['Adj Close_Vix']
data['K_P'] = data['Adj Close_Spy']-data['Adj Close_Vix']

data['S'] = data['Adj Close_Spy']
data['r'] = data['Rate']/100
data['sigma'] = data['Adj Close_Vix']/100 
data['T'] = 1/12
data['q'] = 0.0144

data.pop('Adj Close_Spy')
data.pop('Adj Close_Vix')

data['Call Price'] = data.apply(lambda row: call_black_scholes(
    row['S'], row['K_C'], row['sigma'], row['r'], row['T'], row['q']), axis=1)

data['Put Price'] = data.apply(lambda row: put_black_scholes(
    row['S'], row['K_P'], row['sigma'], row['r'], row['T'], row['q']), axis=1)
print(data)

# Long 100 SPY

# Short one 1-month call OTM option contract on SPY


# Set up the initial columns and flags
data['Call or worthless'] = np.where(data['K_C'] > data['S'].shift(1), 'worthless', 'call')
data['Profit'] = 0


data['S'] = data['S'].fillna(method='ffill')
data['K_C'] = data['K_C'].fillna(method='ffill')

data['Profit'] = np.where(
    data['S'].shift(1) < data['K_C'],  
    100 * data['K_C'].shift(0) - 100 * data['S'].shift(1), 
    0 
)

options_profit = data['Profit'].sum()
print(data[['S', 'K_C', 'Profit']])

print('The total return is: ', options_profit)

# Code Here
data['Profit for Covered Call'] = 0
data['Position_CC'] = 'Hold'

initial_investment = 100 * data['S']

call_itm = data['S'].shift(1) > data['K_C']
call_otm = ~call_itm  

# Worthless option
data['Position_CC'] = np.where(call_otm, 'Sell Next Call', data['Position_CC'])
data['Profit for Covered Call'] = np.where(call_otm, 0, data['Profit for Covered Call']) 

# Sell 100 shares at strick and buy 100 shares at the spot, sell next call
data['Position_CC'] = np.where(call_itm, 'Call Exercised, Repurchase', data['Position_CC'])
data['Profit for Covered Call'] = np.where(call_itm, 100 * data['K_C'] - 100 * data['S'].shift(0), data['Profit for Covered Call'])

data['Monthly Return - Call'] = data['Profit for Covered Call'] / initial_investment

overall_return = (data['Profit for Covered Call'] / initial_investment)*100 

data['Covered Call % Return'] = overall_return


# Display the final strategy outcomes
print(data[['S', 'K_C', 'Profit for Covered Call', 'Position_CC', 'Covered Call % Return']])

print('The returns for the covered call strategy is: ', data['Profit for Covered Call'].sum())


# Code Here
data['Position'] = 'Hold'
data['Profit for Collar'] = 0

call_itm = data['S'].shift(1) > data['K_C']
put_itm = data['S'].shift(1) < data['K_P']
both_otm = ~call_itm & ~put_itm

data['Position'] = np.where(both_otm, 'Repeat Collar', data['Position'])
data['Profit'] = np.where(both_otm, 0, data['Profit'])  

initial_investment = 100 * data['S'].iloc[0]

# Call expires in the money, sell stock at K OR if put expires, sell stock at K. And buy 100 shares at S
# broken down into call expiring and put expiring

# Call
data['Position'] = np.where(call_itm & ~put_itm, 'Call Exercised, Repurchase', data['Position'])
data['Profit for Collar'] = np.where(call_itm & ~put_itm, 100 * data['K_C'].shift(-1) - 100 * data['S'].shift(0), data['Profit for Collar'])

# Put
data['Position'] = np.where(put_itm & ~call_itm, 'Put Exercised, Repurchase', data['Position'])
data['Profit for Collar'] = np.where(put_itm & ~call_itm, 100 * data['K_P'] - 100 * data['S'].shift(0), data['Profit for Collar'])

total_profit_collar = data['Profit for Collar'].sum()
data['Monthly Return - Collar'] = data['Profit for Collar'] / initial_investment

overall_return = (data['Profit for Collar'] / initial_investment)*100 

data['Collar % Return'] = overall_return



print(data[['S', 'K_C', 'K_P', 'Profit for Collar', 'Position', 'Collar % Return']])

print('The profit for the Cost neutral collars strategy is: ', data['Profit for Collar'], '\n')

data['Profit for Collar'].sum()


long_pos = 100 * data['S'].iloc[0]
long_pos_final = 100 * data['S'].iloc[-1]

profit_BH = long_pos_final - long_pos

profit_BH

monthly_return_BH = ((profit_BH / long_pos) / 12)*100

data['Monthly Return - Long only'] = monthly_return_BH

data['Monthly Return - Long only']
results_df = pd.DataFrame({
    'Covered Call % Return': data['Covered Call % Return'],
    'Collar % Return': data['Collar % Return'],
    'Long only % Return':data['Monthly Return - Long only']
})


print("Results DataFrame:")
print(results_df)


'''

There are a couple ways to approach ranking these strategies, most of them changing on what your investment horizon, goals are among other factors. 

Here I will assume a 12-month lock-in where money is reinvested. Thus, we can rely on absolute returns as a metric.

'''

results_df['CS_collar'] = results_df['Cumulative Collar % Return'] = results_df['Collar % Return'].cumsum()
results_df['CS_call'] = results_df['Cumulative Covered Call % Return'] = results_df['Covered Call % Return'].cumsum()
results_df['CS_long'] = results_df['Cumulative Long only % Return'] = results_df['Long only % Return'].cumsum()

collar_max = results_df['CS_collar'].iloc[-1]
call_max = results_df['CS_call'].iloc[-1]
long_max = results_df['CS_long'].iloc[-1]

chosen_one = max(collar_max, call_max, long_max)




print('The chosen strategy is the long-only strategy: ', long_max)

