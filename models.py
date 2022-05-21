from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from performance import Performance
import numpy as np

class ModelManager:
    def __init__(self, train_dataset, test_dataset, target_variable_name):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.target_variable_name = target_variable_name

    def get_data_for_training(self):
        # drop target variable from dataset
        Train_x = self.train_dataset.dataset.drop(self.target_variable_name, axis=1).values.tolist()
        Train_y = self.train_dataset.dataset[self.target_variable_name].values.tolist()

        # split dataset into training and test set
        X_train, X_test, y_train, y_test = train_test_split(Train_x, Train_y, test_size=0.2, random_state=0)
        
        X_train, y_train =  self.train_dataset.pre_process.balance_dataset(X_train, y_train, X_test, y_test)


        # Normalize data
        # from sklearn.preprocessing import StandardScaler
        # sc = StandardScaler()
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)
        

        return X_train, X_test, y_train, y_test
    
    def get_performance(self, model, X_test, y_test, probs):
        y_pred = model.predict(X_test)

        performance = Performance(y_test, y_pred)
        performance.roc_and_pr(y_test, probs) 
        performance.display_confusion_matrix(model, X_test, y_test)
        
        return performance.get_performance()
    
    def train_different_models(self):
        # get data for training
        X_train, X_test, y_train, y_test = self.get_data_for_training()
        
        from sklearn.model_selection import cross_val_score

        models = [RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0), KNeighborsClassifier(), GaussianNB(), LogisticRegression()]
        
        for model in models:
            print("Training model: " + str(model))
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)
            print(self.get_performance(model, X_test, y_test, probs))

            scores = cross_val_score(model, X_train, y_train, cv = 5, scoring='accuracy')
            print('Cross-validation scores: {}'.format(scores))
            print('Average cross-validation score: {}'.format(scores.mean()))

        return model

    
