import numpy as np
from sklearn import preprocessing

# import data
with open('data_set_wwm_modified_context_vec_210809_train_feature_168647.txt') as f1:  # normalized data
    feature = f1.readlines()
f1.close()
with open('data_set_wwm_modified_context_next_word_vec_210809_train_label_168647.txt') as f2:
    label = f2.readlines()
f2.close()

data_set_for_feature = [eval(line.strip('\n')) for line in feature]  # independent variable
data_set_for_label = [eval(line.strip('\n')) for line in label]  # dependent variable
# change to numpy format
data_set_for_feature = np.array(data_set_for_feature)
data_set_for_label = np.array(data_set_for_label)
# data normalization
data_set_for_feature = preprocessing.scale(data_set_for_feature)
# data_set_for_label = preprocessing.scale(data_set_for_label)

