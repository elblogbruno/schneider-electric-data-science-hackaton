from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier

# 4 Calcular performance de los modelos
#### 4.1 F-Score
#### 4.2 Precision
#### 4.3 Recall
#### 4.4 Accuracy


## Recall --> tp / (tp + fn) --> tp = true positives, fn = false negatives
## Precision --> tp / (tp + fp) --> tp = true positives, fp = false positives
## f1_score --> 2 * (precision * recall) / (precision + recall)
## Accuracy --> (tp + tn) / (tp + tn + fp + fn) --> tp = true positives, tn = true negatives, fp = false positives, fn = false negatives


class Performance:
    def __init__(self, dataset=None, model=None):
        self.dataset = dataset
        self.model = model

    def calculate_f1_score(self, y_true, y_pred, avg):
        return f1_score(y_true, y_pred, average=avg)
    
    def calculate_precision_score(self, y_true, y_pred, avg):
        return precision_score(y_true, y_pred, average=avg)
    
    def calculate_recall_score(self, y_true, y_pred, avg):
        return recall_score(y_true, y_pred, average=avg)
    
    def calculate_accuracy_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def get_performance(self, y_true, y_pred, avg=None):
        return {
            'f1_score': f1_score(y_true, y_pred, average='micro'),
            'precision': precision_score(y_true, y_pred, average='micro'),
            'recall': recall_score(y_true, y_pred, average='micro'),
            'accuracy': accuracy_score(y_true, y_pred)
        }
        
    
    
# y_true = [0, 1, 2, 0, 1, 2]
# y_pred = [0, 2, 1, 0, 0, 1]
# p = Performance()

# a = p.calculate_accuracy_score(y_true, y_pred)
# print(a)

    
    
    