#%% IMPORT
import pandas as pd
import matplotlib.pyplot as plt
from business import (proceeding_EDA,
                      NumberDataExploration,
                      CharDataExploration,
                      DataExploration)
pd.set_option('display.max_columns', None)

df = pd.read_csv('train.csv')

df.head()
# df.info()
# df.describe()

#%% EDA
proceeding_EDA(DE_process=NumberDataExploration, 
               dataframe=df, 
               col_name='Compartments')

proceeding_EDA(DE_process=NumberDataExploration, 
               dataframe=df, 
               col_name='Weight Capacity (kg)')

proceeding_EDA(DE_process=NumberDataExploration, 
               dataframe=df, 
               col_name='Price')

proceeding_EDA(DE_process=CharDataExploration, 
               dataframe=df, 
               col_name='Material')

proceeding_EDA(DE_process=CharDataExploration, 
               dataframe=df, 
               col_name='Size')

proceeding_EDA(DE_process=CharDataExploration, 
               dataframe=df, 
               col_name='Waterproof')

proceeding_EDA(DE_process=CharDataExploration, 
               dataframe=df, 
               col_name='Style')

proceeding_EDA(DE_process=CharDataExploration, 
               dataframe=df, 
               col_name='Color')

proceeding_EDA(DE_process=CharDataExploration, 
               dataframe=df, 
               col_name='Brand')
