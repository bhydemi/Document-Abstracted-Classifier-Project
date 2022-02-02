from sklearn.metrics import  accuracy_score,precision_recall_fscore_support, cohen_kappa_score, make_scorer
import pandas as pd

def performance_measure_classification(x_test, y_test, model, model_name):
    y_pred = model.predict(x_test)
    performance_dict = {}
    f1_score =  make_scorer(accuracy_score)
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_pred, y_test, average='weighted')
    performance_dict['accuracy'] = accuracy_score(y_pred, y_test)
    performance_dict['cohen_kappa_score'] = cohen_kappa_score(y_pred, y_test)
    performance_dict['recall'] = model_recall
    performance_dict['precision'] = model_precision
    performance_dict['f1'] = model_f1
    return pd.DataFrame(performance_dict, index=[model_name])
