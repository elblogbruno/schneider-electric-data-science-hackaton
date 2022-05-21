from sklearn.ensemble import RandomForestClassifier

import numpy as np

def train_model(train_dataset):
    print("Training model...")
    # train_dataset.pre_process.drop_columns(['targetRelease'])
    target_variable_name = 'pollutant'

    # drop target variable from dataset
    Train_x = train_dataset.dataset.drop(target_variable_name, axis=1).values.tolist()
    Train_y = train_dataset.dataset[target_variable_name].values.tolist()

    # split dataset into training and test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(Train_x, Train_y, test_size=0.2, random_state=0)

    model = RandomForestClassifier(n_jobs=1, oob_score=True, n_estimators=10)
    model.fit(X_train, y_train)

    return model, np.array(X_test), np.array(y_test)
