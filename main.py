#%% IMPORT
import pandas as pd
from business import proceeding_EDA, NumberDataExploration, CharDataExploration
pd.set_option('display.max_columns', None)

df = pd.read_csv('train.csv')
# df.head()
# df.info()
# df.describe()

#%% EDA
proceeding_EDA(NumberDataExploration, df, 'Compartments')

proceeding_EDA(CharDataExploration, df, 'Brand')