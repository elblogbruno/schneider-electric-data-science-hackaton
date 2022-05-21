import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd

class PrepareDataset:
    def __init__(self, dataset, name='dataset'):
        self.dataset = dataset
        self.name = name

    def preprocess_dataset(self):
        self.dataset = self._preprocess_dataset()
        return self.dataset

    def _get_dataset_balance(self, dataset):
        """
        Returns the balance of the dataset
        """
        if 'pollutant' not in dataset.columns:
            print('test data received')
            return None
        return dataset.groupby('pollutant').size()

    def _get_nan_columns(self, dataset):
        """
        Returns the columns with missing values in the dataset and the number of missing values
        """
        columns = dataset.columns[dataset.isnull().any()]
        missing_values = dataset.isnull().sum()

        # calculate the percentage of missing values for each column
        missing_values_percentage = (missing_values / dataset.shape[0]) * 100

        print('Columns with missing values:')
        print(columns)
        print('Number of missing values:')
        print(missing_values)
        print('Percentage of missing values:')
        print(missing_values_percentage)

        has_nan_columns = missing_values.any()

        return has_nan_columns

    def _fix_nan_columns(self, dataset):
        """
        Fix the missing values in the dataset
        """
        # replace the missing values with the mean of the column
        dataset.fillna(dataset.mean(), inplace=True)

        return dataset

    def fix_columns_type(self):
        """
        Fix the columns type to numeric
        """
        numeric_columns = ['max_temp', 'max_wind_speed', 'min_temp', 'min_wind_speed', 'reportingYear', 'avg_temp', 'avg_wind_speed', 'MONTH', 'DAY', 'DAY WITH FOGS']

        for column in numeric_columns:
            self.dataset[column] = pd.to_numeric(self.dataset[column], errors='coerce')

        return self.dataset


    def get_categorical_columns(self):
        """
        Returns the columns that values are not numbers 
        """
        col = self.dataset.select_dtypes(include='object').columns
        return col

    def _get_category_count(self, column):
        """
        Returns the number of categories in each column
        """
        return self.dataset[column].nunique()

    def _get_columns_category(self, column):
        """
        Returns the categories of the column
        """
        return self.dataset[column].unique()

    def categorize_data(self, columns_to_categorize):
        """
        Categorize the data in the columns specified
        """
        for column in columns_to_categorize:
            print('Column: {0} Size: {1}'.format(column, self._get_category_count(column)))
            column_categories = self._get_columns_category(column)
            print(len(column_categories))

            # for each category assign a number to it and replace the category with the number
            for i, category in enumerate(column_categories):
                self.dataset[column] = self.dataset[column].replace(category, i)
                

        return self.dataset

    def get_highest_correlation(self, n=10):
        """
        Returns the n highest correlation variables
        """
        return self.dataset.corr().nlargest(n, 'pollutant')

    def show_dataset_correlation_heatmap(self, ):
        """
        Returns the correlation between the variables
        """
        try:
            # set columns type to numeric to calculate the correlation
            coor = self.dataset.select_dtypes(include=['number']).corr()
            
            # remove null values
            coor = coor.dropna(how='all')

            # show the heatmap of the correlation
            sns.heatmap(coor, annot=True)
            plt.title('Correlation Heatmap {0}'.format(self.name))
            plt.show()
        except Exception as e:
            print(e)
            pass
        
    def _preprocess_dataset(self):
        print("Checking datase Balance")
        print(self._get_dataset_balance(self.dataset))
        
        print("Checking dataset Nan Columns")
        has_nan_columns = self._get_nan_columns(self.dataset)

        if has_nan_columns:
            print("Fixing dataset Nan Columns...")
            self.dataset = self._fix_nan_columns(self.dataset)

        self.fix_columns_type()

        # # get columns that are strings and categorical
        # categorical_columns = self.get_categorical_columns()
        # self.categorize_data(categorical_columns)     

        return self.dataset