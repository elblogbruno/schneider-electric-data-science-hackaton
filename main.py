from cgi import test
import pandas as pd
from dataset import Dataset
import os
from models import ModelManager

from store_models import StoreModels

# TAREAS
# 1. Cargar Datos 
    # 1.1 Cargar Datos de los PDF 
    # 1.2 Cargar Datos de los Excel (Hecho)
    # 1.3 Cargar Datos de los CSV (Hecho)

# 2. Procesar Datos    
    # 2.1 Ver si el dataset tiene alguna columna con valores nulos o vacios [HECHO]
    # 2.4 Ver si el dataset esta desbalanceado (LO ESTA) [HECHO]
        # 2.4.1 Arreglarlo 
    # 2.5 Ver si hay que categorizar los datos 
    # 2.6 Ver si hay que normalizar los datos
    # 2.7 Ver variables inutiles (correlacion entre variables) 
    
    
# 3 Entrenar Modelos
    # KNN [HECHO]
    # SVM
    # Random Forest [HECHO]
    # Naive Bayes [HECHO]
    # Decision Tree
    # Logistic Regression [HECHO]

# 4 Calcular performance de los modelos
    # 4.1 F1-Score [HECHO]
    # 4.2 Precision [HECHO]
    # 4.3 Recall [HECHO]
    # 4.4 Accuracy [HECHO]

# 5 Guardar Modelos 
    # 5.1 Guardar Graficas
        # 5.2 ROC Curve [HECHO]
        # 5.3 Precision-Recall Curve [HECHO]
        # 5.4 Confusion Matrix
    # 5.2 GUARDAR .CSV amb els resultats de predir per a cadascuna de les files de test_x.csv

# pollutant: Type of pollutant emitted (Target variable). In order to follow the same standard, you must encode this variables as follows:

# pollutant	number
# Nitrogen oxides (NOX)	0
# Carbon dioxide (CO2)	1
# Methane (CH4)	2



def load_and_process_data(target_variable_name, force = False):
    # headers_to_remove = ["REPORTER NAME",  "FacilityInspireID", 'facilityName', 'targetRelease', 'MONTH', 'DAY', 'CONTINENT', 'max_wind_speed', 'min_wind_speed', 'avg_wind_speed', 'max_temp', 'min_temp', 'City']
    headers_to_remove = ['targetRelease', 'MONTH', 'DAY', 'CONTINENT', 'max_wind_speed', 'min_wind_speed', 'avg_wind_speed', 'max_temp', 'min_temp', 'avg_temp']

    if os.path.exists('result/train_processed.csv') == False or force:
        print("Loading dataset...")
        dataset_csv_1 = Dataset(type='csv', file_name='train/train1.csv', target_variable_name=target_variable_name)
        dataset_csv_1.remove_headers(headers_to_remove)

        headers = dataset_csv_1.get_headers()

        dataset_csv_2 = Dataset(type='csv', file_name='train/train2.csv', sep=';', target_variable_name=target_variable_name)
        dataset_csv_2.remove_headers(headers_to_remove)
        
        dataset_json_1 = Dataset(type='json', file_name='first', target_variable_name=target_variable_name)
        headers_json = dataset_json_1.get_headers()
        dataset_json_2 = Dataset(type='json', file_name='second', target_variable_name=target_variable_name)
        dataset_json_3 = Dataset(type='json', file_name='third', target_variable_name=target_variable_name)

        dataset_pdf = Dataset(type='pdf', target_variable_name=target_variable_name)
        dataset_pdf.remove_headers(headers_to_remove)
        dataset_pdf.process_data()


        # delete headers that are not in dataset_json_1 and dataset_json_2
        same_headers = [header for header in headers_json if header in headers]
        diff_headers = [header for header in headers_json if header not in headers]

        # diff_headers.append(headers_to_remove)
        print(same_headers)
        print(diff_headers)

        dataset_json_1.remove_headers(diff_headers)
        dataset_json_2.remove_headers(diff_headers)
        dataset_json_3.remove_headers(diff_headers)
        
        dataset_csv_1.process_data()
        dataset_csv_2.process_data()

        dataset_json_1.process_data()
        dataset_json_2.process_data()
        dataset_json_3.process_data()

        train_dataset = Dataset(child_datasets=[dataset_csv_1, dataset_csv_2, dataset_json_1, dataset_json_2, dataset_json_3, dataset_pdf], target_variable_name=target_variable_name)
        
        # get columns that are strings and categorical
        categorical_columns = train_dataset.pre_process.get_categorical_columns()
        train_dataset.pre_process.categorize_data(categorical_columns)     

        train_dataset.save_dataset(file_name='result/train_processed.csv')
        print(train_dataset.pre_process.get_dataset_balance(train_dataset.dataset))
    

    else:
        print("Loading dataset from folder...")
        train_dataset = Dataset(type='csv', file_name='result/train_processed.csv', target_variable_name=target_variable_name)
        
    if os.path.exists('result/test_processed.csv') == False or force:
        test_dataset = Dataset(type='csv', file_name='test/test_x.csv', target_variable_name=target_variable_name)
        
        test_dataset.remove_headers(['EPRTRAnnexIMainActivityCode', 'EPRTRSectorCode', 'test_index'])
        
        test_dataset.remove_headers(headers_to_remove)
        
        test_dataset.process_data()

        categorical_columns = test_dataset.pre_process.get_categorical_columns()
        test_dataset.pre_process.categorize_data(categorical_columns)      # categorical_columns = ['pollutant']
        
        test_dataset.save_dataset(file_name='result/test_processed.csv')
    else:
        test_dataset = Dataset(type='csv', file_name='result/test_processed.csv', target_variable_name=target_variable_name)

    return train_dataset, test_dataset
    



if __name__ == '__main__':
    target_variable_name = 'pollutant'

    train_dataset, test_dataset = load_and_process_data(target_variable_name, force= True)

    print(len(train_dataset.dataset))
    print(len(test_dataset.dataset))

    train_dataset.pre_process.show_dataset_correlation_heatmap()

    highest_corrl = train_dataset.pre_process.get_highest_correlation()


    model = ModelManager(train_dataset, test_dataset, target_variable_name)
    model.train_different_models()

    ##### TESTING ZONE #####
    
    # SAVE DATASET
    # file = "models.csv"
    # sm = ManageModels()
    # sm.store_model(model, file)
    
    


    






