import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataExploration:
    def __init__(self, dataframe, col_name):
        self.__col_name = col_name
        self.data = dataframe[col_name]
        self.__data_length = len(self.data)
        
    def check_cleaning_status(self):
        
        null_count = self.data.isnull().sum()
        filling_percentage = (self.__data_length - null_count) / self.__data_length
        
        return f"'{self.__col_name}': {self.__data_length} observation,\
 {filling_percentage * 100:.2f}% available."
 
    # For char data 
    def analyze_char_stat(self):
        return self.data.value_counts(normalize=True)
    
    def plot_pie_chart(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(self.data.value_counts(),
            labels=self.data.value_counts().index,
            autopct = '%1.1f%%'
            )
        plt.show()
        
    # For number data
    def analyze_number_stat(self):
        mean = self.data.mean()
        std = self.data.std()
        lowest = self.data.min()
        highest = self.data.max()
        return f"Data ranges from {lowest} to {highest} with mean of {mean} and standard deviation of {std}"
        
    def plot_histogram(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(self.data)
        plt.show()
        
    
class NumberDataExploration(DataExploration):
    def __init__(self, dataframe, col_name):
        super().__init__(dataframe, col_name)
        
    def check_cleaning_status(self):
        parent_print = super().check_cleaning_status()
        current_print = parent_print + " Data type of number."
        print(current_print)
    
    def analyze_statistics(self):
        statement = super().analyze_number_stat()
        print(statement)
    
    def plot_statistics(self):
        super().plot_histogram()
    
class CharDataExploration(DataExploration):
    def __init__(self, dataframe, col_name):
        super().__init__(dataframe, col_name)
        
    def check_cleaning_status(self):
        parent_print = super().check_cleaning_status()
        current_print = parent_print + " Data type of character."
        print(current_print)
    
    def analyze_statistics(self):
        result = super().analyze_char_stat()
        print(result)
    
    def plot_statistics(self):
        super().plot_pie_chart()
    
def proceeding_EDA(DE_process, dataframe, col_name):
    de = DE_process(dataframe, col_name)
    de.check_cleaning_status()
    de.analyze_statistics()
    de.plot_statistics()