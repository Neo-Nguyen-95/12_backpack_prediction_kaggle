#%% LIBRARY
# Data part
import pandas as pd
import numpy as np
from business import proceeding_EDA, NumberDataExploration, CharDataExploration
pd.set_option('display.max_columns', None)

# Machine learning part
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


#%% IMPORT

df = pd.read_csv('train.csv')

df.head()
df.info()
# df.describe()

df_no_null = df.dropna()
df_no_null.info()

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

#%% CORR CHECK
correlation = df[['Compartments', 'Weight Capacity (kg)', 'Price']].corr()
print(correlation)

#%% PREPROCESSOR

numeric_pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
    )

categorical_pipeline = make_pipeline(
    SimpleImputer(
        # strategy='most_frequent',
        strategy='constant',
        fill_value='missing'
        ),
    OneHotEncoder()
    )

num_features = ['Compartments', 'Weight Capacity (kg)']
cat_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 
                'Waterproof', 'Style', 'Color']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, num_features),
        ('cat', categorical_pipeline, cat_features)
        ]
    )

#%% MODEL 1: FULL
target = 'Price'
y = df[target]

X = df.drop([target, 'id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

model_full = make_pipeline(
    preprocessor,
    Ridge()
    )

# Model trained with full data
model_full.fit(X_train, y_train)

y_pred_train = model_full.predict(X_train)
y_pred_val = model_full.predict(X_test)

# Baseline
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
print(f'Baseline MAE: {mae_baseline}')

rmse_baseline = np.sqrt(mean_squared_error(y_train, y_pred_baseline))
print(f'Baseline RMSE: {rmse_baseline}')

# Model with Train data
mae_train = mean_absolute_error(y_train, y_pred_train)
print(f'Train MAE: {mae_train}')

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f'Train RMSE: {rmse_baseline}')

# Model with Validate data
mae_val = mean_absolute_error(y_test, y_pred_val)
print(f'Validate MAE: {mae_val}')

rmse_val = np.sqrt(mean_squared_error(y_test, y_pred_val))
print(f'Validate RMSE: {rmse_val}')

#%% MODEL 2: Model trained with no-null data
target = 'Price'
y = df_no_null[target]

X = df_no_null.drop([target, 'id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

model_no_null = make_pipeline(
    preprocessor,
    Ridge()
    )

model_no_null.fit(X_train, y_train)
y_pred_train = model_no_null.predict(X_train)
y_pred_val = model_no_null.predict(X_test)

# Baseline
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
print(f'Baseline MAE: {mae_baseline}')

rmse_baseline = np.sqrt(mean_squared_error(y_train, y_pred_baseline))
print(f'Baseline RMSE: {rmse_baseline}')

# Model with Train data
mae_train = mean_absolute_error(y_train, y_pred_train)
print(f'Train MAE: {mae_train}')

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f'Train RMSE: {rmse_baseline}')

# Model with Validate data
mae_val = mean_absolute_error(y_test, y_pred_val)
print(f'Validate MAE: {mae_val}')

rmse_val = np.sqrt(mean_squared_error(y_test, y_pred_val))
print(f'Validate RMSE: {rmse_val}')

#%% SUBMISSION

df_test = pd.read_csv('test.csv')
df_test.info()
X_sub = df_test.drop('id', axis=1)
y_pred_sub = model_full.predict(X_sub)
df_sub = df_test[['id']]
df_sub['Price'] = y_pred_sub
df_sub.set_index('id', inplace=True)

# Subsitute with better prediction from no_null model
df_test_no_null = df_test.dropna()
df_test_no_null.info()
X_no_null = df_test_no_null.drop('id', axis=1)
y_no_null_pred = model_no_null.predict(X_no_null)

df_upgrade = df_test_no_null[['id']]
df_upgrade['Price'] = y_no_null_pred

df_upgrade.set_index('id', inplace=True)

df_sub.update(df_upgrade)


df_sub.to_csv('Neo_submission.csv', index=True)

#%% sample check
df_sample = pd.read_csv('sample_submission.csv')
