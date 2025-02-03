import pandas as pd

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
 
    def plot_pie_chart(self):
        ...
        
    def plot_histogram(self):
        ...
        
    
    
class NumberDataExploration(DataExploration):
    def __init__(self, dataframe, col_name):
        super().__init__(dataframe, col_name)
        
    def check_cleaning_status(self):
        parent_print = super().check_cleaning_status()
        current_print = parent_print + " Data type of number."
        print(current_print)
    
    def analyze_statistics(self):
        pass
    
    def plot_statistics(self):
        pass
    
class CharDataExploration(DataExploration):
    def __init__(self, dataframe, col_name):
        super().__init__(dataframe, col_name)
        
    def check_cleaning_status(self):
        parent_print = super().check_cleaning_status()
        current_print = parent_print + " Data type of character."
        print(current_print)
    
    def analyze_statistics(self):
        pass
    
    def plot_statistics(self):
        pass
    
def proceeding_EDA(DE_process, dataframe, col_name):
    de = DE_process(dataframe, col_name)
    de.check_cleaning_status()
    de.analyze_statistics()
    de.plot_statistics()