from cgi import test
import pandas as pd
from dataset import Dataset
import os
from models import train_model
from performance import Performance

# TAREAS
# 1. Cargar Datos 
    # 1.1 Cargar Datos de los PDF 
    # 1.2 Cargar Datos de los Excel (Hecho)
    # 1.3 Cargar Datos de los CSV (Hecho)

# 2. Procesar Datos    
    # 2.1 Ver si el dataset tiene alguna columna con valores nulos o vacios (Hecho)
    # 2.4 Ver si el dataset esta desbalanceado (LO ESTA) (Hecho)
        # 2.4.1 Arreglarlo 
    # 2.5 Ver si hay que categorizar los datos
    # 2.6 Ver si hay que normalizar los datos
    # 2.7 Ver variables inutiles (correlacion entre variables) 
    
    
# 3 Entrenar Modelos
     # KNN
        # SVM
        # Random Forest
        # Naive Bayes
        # Decision Tree
    # Logistic Regression

# 4 Calcular performance de los modelos
    # 4.1 F1-Score
    # 4.2 Precision
    # 4.3 Recall
    # 4.4 Accuracy

# 5 Guardar Modelos 
    # 5.1 Guardar Graficas

# pollutant: Type of pollutant emitted (Target variable). In order to follow the same standard, you must encode this variables as follows:

# pollutant	number
# Nitrogen oxides (NOX)	0
# Carbon dioxide (CO2)	1
# Methane (CH4)	2



def load_and_process_data():
    if os.path.exists('result/train_processed.csv') == False:
        print("Loading dataset...")
        dataset_csv_1 = Dataset(type='csv', file_name='train/train1.csv')
        headers = dataset_csv_1.get_headers()
        dataset_csv_2 = Dataset(type='csv', file_name='train/train2.csv', sep=';')
        
        dataset_json_1 = Dataset(type='json', file_name='first')
        headers_json = dataset_json_1.get_headers()
        dataset_json_2 = Dataset(type='json', file_name='second')
        dataset_json_3 = Dataset(type='json', file_name='third')

        # delete headers that are not in dataset_json_1 and dataset_json_2
        same_headers = [header for header in headers_json if header in headers]
        diff_headers = [header for header in headers_json if header not in headers]
        
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

        train_dataset = Dataset(child_datasets=[dataset_csv_1, dataset_csv_2, dataset_json_1, dataset_json_2, dataset_json_3])
        
        # get columns that are strings and categorical
        categorical_columns = train_dataset.pre_process.get_categorical_columns()
        train_dataset.pre_process.categorize_data(categorical_columns)     

        train_dataset.save_dataset(file_name='result/train_processed.csv')

    else:
        print("Loading dataset from folder...")
        train_dataset = Dataset(type='csv', file_name='result/train_processed.csv')
        
    if os.path.exists('result/test_processed.csv') == False:
        test_dataset = Dataset(type='csv', file_name='test/test_x.csv')
        test_dataset.remove_headers(['EPRTRAnnexIMainActivityCode', 'EPRTRSectorCode', 'test_index'])
        test_dataset.process_data()

        categorical_columns = test_dataset.pre_process.get_categorical_columns()
        test_dataset.pre_process.categorize_data(categorical_columns)      # categorical_columns = ['pollutant']
        
        test_dataset.save_dataset(file_name='result/test_processed.csv')
    else:
        test_dataset = Dataset(type='csv', file_name='result/test_processed.csv')

    return train_dataset, test_dataset
    



if __name__ == '__main__':
    train_dataset, test_dataset = load_and_process_data()

    test_dataset.remove_headers(['EPRTRAnnexIMainActivityCode', 'EPRTRSectorCode', 'test_index'])
    
    model, X_test, y_test = train_model(train_dataset)
    
    y_pred = model.predict(X_test)

    performance = Performance(test_dataset.dataset, model)

    print(performance.get_performance(y_test, y_pred))
    






