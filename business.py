# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import numpy as np

#%% BASE EDA CLASS
class DataExploration:
    def __init__(self, dataframe, col_name):
        self._col_name = col_name
        self.data = dataframe[col_name]
        self._data_length = len(self.data)
    
    # Check null variable
    def check_cleaning_status(self):
        null_count = self.data.isnull().sum()
        filling_percentage = (self._data_length - null_count) / self._data_length
        
        return {
            "column": self._col_name,
            "total_observation": self._data_length,
            "filling_percentage": filling_percentage
            }
 
    # For char data 
    def analyze_char_stat(self):
        return self.data.value_counts(normalize=True)

    def plot_pie_chart(self):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
        ax.pie(self.data.value_counts(),
            labels=self.data.value_counts().index,
            autopct = '%1.1f%%'
            )
        plt.title(f'Distribution of {self._col_name}')
        plt.show()

    # For number data
    def analyze_number_stat(self):
        mean = self.data.mean()
        std = self.data.std()
        lowest = self.data.min()
        highest = self.data.max()
        return f"Data ranges from {lowest:.2f} to {highest:.2f} with mean of {mean:.2f} and standard deviation of {std:.2f}.\n"
        
    def plot_histogram(self):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        sns.histplot(self.data)
        plt.title(f'Distribution of {self._col_name}')
        plt.show()
        
#%% EDA SPECIALIZED FOR NUMBER TYPE ANALYSIS  
class NumberDataExploration(DataExploration):
    def __init__(self, dataframe, col_name):
        super().__init__(dataframe, col_name)
        
    def check_cleaning_status(self):
        status = super().check_cleaning_status()
        status['data_type'] = "Number"
        return status

    def analyze_statistics(self):
        statement = super().analyze_number_stat()
        return statement
    
    def plot_statistics(self):
        super().plot_histogram()
        plt.show()

#%% EDA SPECIALIZED FOR CHAR TYPE ANALYSIS
class CharDataExploration(DataExploration):
    def __init__(self, dataframe, col_name):
        super().__init__(dataframe, col_name)
        
    def check_cleaning_status(self):
        status = super().check_cleaning_status()
        status['data_type'] = "Character"
        return status

    def analyze_statistics(self):
        result = super().analyze_char_stat()
        return result
    
    def plot_statistics(self):
        super().plot_pie_chart()
        plt.show()

#%% PLUG IN FUNCTION
def proceeding_EDA(DE_process, dataframe, col_name):
    de = DE_process(dataframe, col_name)
    if hasattr(de, "check_cleaning_status"):
        print(de.check_cleaning_status())
        
    if hasattr(de, "analyze_statistics"):
        print(de.analyze_statistics())
        
    if hasattr(de, "plot_statistics"):
        de.plot_statistics()
    