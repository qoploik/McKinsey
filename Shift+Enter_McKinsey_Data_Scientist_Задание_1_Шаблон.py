### General import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read input file
df = pd.read_csv('opsd_austria_daily.csv', index_col=0)
df.index = pd.to_datetime(df.index)

# Task 1a: visualize distribution & time changes of input data
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
# Distribution of electricity consumption & wind production
ax[0,0].hist(..., bins=50, color='palegreen', label='consumption') # put here electricity consumption values
ax[0,0].set_xlabel('Consumption and wind power production, GWh')
ax[0,0].set_ylabel('Count')
ax2 = ... # insert the double y-axis using twinx to display wind power production
ax2.hist(..., bins=50, color='lightblue', label='wind') # put here wind power production values
ax2.set_ylabel('Count')
ax[0,0].legend(loc='best')
ax2.legend(loc='best')
# Distribution of solar power production
ax[0,1].hist(..., bins=50, color='darksalmon') # put here solar power production values
ax[0,1].set_xlabel('Solar power production, GWh')
ax[0,1].set_ylabel('Count')
# Time series
ax[1,0].plot(..., linewidth = 0.5, label='consumption') # put here electricity consumption values
ax[1,0].plot(..., linewidth = 0.5, label='wind+solar production') # put here merged wind + solar data values
ax[1,0].set_xlabel('Time')
ax[1,0].set_ylabel('Energy, GWh')
ax[1,0].legend(loc='best')
# Price distribution
ax[1,1].plot(..., linewidth = 0.5) # substitute ... with spot price from the dataset
ax[1,1].set_xlabel('Time')
ax[1,1].set_ylabel('Price, Euro/1 kWh')
ax[1,1].legend(loc='best')
plt.show()

# Task 1b: analysing the seasonality for 2019 year
''' defining a rolling average '''
df['Electricity_consumption_RA'] = ...
df['Wind_production_RA'] = ...
df['Solar_production_RA'] = ...

# Plot
...