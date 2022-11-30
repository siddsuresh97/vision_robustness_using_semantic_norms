import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
def calculate_metrics(target, pred, dataset_type, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/{}_precision'.format(dataset_type): precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/{}_recall'.format(dataset_type): recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/{}_f1'.format(dataset_type): f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/{}_precision'.format(dataset_type): precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/{}_recall'.format(dataset_type): recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/{}_f1'.format(dataset_type): f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/{}_precision'.format(dataset_type): precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/{}_recall'.format(dataset_type): recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/{}_f1'.format(dataset_type): f1_score(y_true=target, y_pred=pred, average='samples'),
            'weighted/{}_precision'.format(dataset_type): precision_score(y_true=target, y_pred=pred, average='weighted'),
            'weighted/{}_recall'.format(dataset_type): recall_score(y_true=target, y_pred=pred, average='weighted'),
            'weighted/{}_f1'.format(dataset_type): f1_score(y_true=target, y_pred=pred, average='weighted')
            }