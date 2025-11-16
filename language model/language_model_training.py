import time
import pickle
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import multithread_train_data_loading


def ridge_model_training(dimension, data_set_for_feature, data_set_for_label):
    for d in dimension:
        time_start = time.time()
        # select data
        data_set_for_label_ndimension = [i[d] for i in data_set_for_label]
        # generate model
        model = Ridge()
        alpha_can = np.logspace(-3, 4, 22)
        ridge_model = GridSearchCV(model, param_grid = {'alpha': alpha_can}, cv = 4, scoring = 'neg_mean_squared_error')
        ridge_model.fit(data_set_for_feature, data_set_for_label_ndimension)
        # save model
        output = open('pred_ridge_model_new_20201221/' + str(d + 1) + 'd_model.pkl', 'wb')
        pickle.dump(ridge_model, output)
        output.close()
        print('model {} cost time: {}'.format(str(d + 1), time.time() - time_start)) 

def SVR_model_training(dimension, data_set_for_feature, data_set_for_label):
    for d in dimension:
        time_start = time.time()
        # select data
        data_set_for_label_ndimension = [i[d] for i in data_set_for_label]
        # generate model
        model = SVR()
        param_grid = {'kernel':['poly', 'rbf'],
                      'C':[1e-2, 1e-1, 1e0, 1e1, 1e2],
                      'gamma':np.linspace(0.001, 100, 5)}
        SVR_model = GridSearchCV(model, param_grid, cv = 4, scoring = 'neg_mean_absolute_error')
        SVR_model.fit(data_set_for_feature, data_set_for_label_ndimension)
        # save model
        output = open('pred_SVR_model/' + str(d + 1) + 'd_model.pkl', 'wb')
        pickle.dump(SVR_model, output)
        output.close()
        print('model {} cost time: {}'.format(str(d + 1), time.time() - time_start))    


if __name__ == '__main__':
    # model generating
    dimension = list(range(1024))
    ridge_model_training(dimension, multithread_train_data_loading.data_set_for_feature, multithread_train_data_loading.data_set_for_label)

