from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)
import numpy as np


def evaluation_metrics_to_string(y_true, y_pred):
    return _create_output(y_true, y_pred)


def _create_output(y_true, y_pred):
    result = ''
    result += _create_partial_output(mean_squared_error(y_true, y_pred), 'mse')
    result += _create_partial_output(mean_absolute_error(y_true, y_pred), 'mae')
    result += _create_partial_output(np.sqrt(mean_squared_error(y_true, y_pred)), 'rmse')
    return result


def _create_partial_output(metric_value, metric_name):
    return '| %s: %.5f' % (metric_name, metric_value)


def gen_output(predicted_results, true_res):  # this function is used only in the performance prediction script
    tn, fp, fn, tp = confusion_matrix(true_res, predicted_results).ravel()
    output_str = ''
    output_str += 'acc : %.3f | ' % ((tp+tn)/(tp+tn+fp+fn))
    output_str += 'prec correct : %.3f | ' % (tp/(tp+fp))
    output_str += 'rec correct : %.3f | ' % (tp/(tp+fn))
    output_str += 'prec wrong : %.3f | ' % (tn/(tn+fn))
    output_str += 'rec wrong: %.3f ' % (tn/(tn+fp))
    return output_str
