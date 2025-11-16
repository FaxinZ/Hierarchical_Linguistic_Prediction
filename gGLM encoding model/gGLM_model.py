import numpy as np
from sklearn.linear_model import LinearRegression
from nistats.regression import ARModel


def gGLM_modelling(subj_num, group_data, ind_dif_DM, sess_dif_DM, train_DM, test_DM, model_type):
    ''' apply the gGLM to fit brain parcels '''  
    subj_data_r_sq = []
    N = group_data.shape[0]
    for v in range(N):
        group_data_vox = group_data[v, :]
        test_set = np.array(group_data_vox[423 * subj_num : 423 * (subj_num + 1)])
        train_set = np.hstack((group_data_vox[0 : 423 * subj_num], group_data_vox[423 * (subj_num + 1) : group_data_vox.shape[0]]))
        # train set remove baseline - ind_dif
        Ind_model = LinearRegression().fit(ind_dif_DM, train_set)
        train_set = train_set - Ind_model.predict(ind_dif_DM)
        # train set remove baseline - sess_dif
        Sess_model = LinearRegression().fit(sess_dif_DM, train_set)
        train_set = train_set - Sess_model.predict(sess_dif_DM)
        # train set pre-whitening
        whiten_model = ARModel(train_DM, 1)  # AR(1) model
        train_set = whiten_model.whiten(train_set)
        # test set remove baseline
        test_set = test_set - np.mean(test_set)
        # test set remove baseline - sess_dif
        Sess_model1 = LinearRegression().fit(sess_dif_DM[: 423, :], test_set)
        test_set = test_set - Sess_model1.predict(sess_dif_DM[: 423, :])
        # GLM modelling
        GLM_model = LinearRegression().fit(train_DM, train_set)
        r_sq = GLM_model.score(test_DM, test_set)
        subj_data_r_sq.append(r_sq)
        print('ongoing subj: {}; the ongoing number: {}; r_sq: {}'.format(subj_num, v, r_sq))
    np.save(model_type + r'_SUB' + str(subj_num).zfill(3) + '.npy', np.array(subj_data_r_sq))
    return

print('successfully loaded!')



''' Showing how to use '''
if __name__ == '__main__':
    # feature loading
    ind_dif_DM = np.load('Data/ind_dif_DM.npy', mmap_mode = 'r')  # design matrix for individual difference
    sess_dif_DM = np.load('Data/sess_dif_DM.npy', mmap_mode = 'r')  # design matrix for multiple sessions
    train_DM_Wrepr = np.load('Data/ws_seperated/train_DM_Wrepr.npy', mmap_mode = 'r')  # word-level (train)
    test_DM_Wrepr = np.load('Data/ws_seperated/test_DM_Wrepr.npy', mmap_mode = 'r')  # word-level (test)
    train_DM_Srepr = np.load('Data/ws_seperated/train_DM_Srepr.npy', mmap_mode = 'r')  # sentence-level (train)
    test_DM_Srepr = np.load('Data/ws_seperated/test_DM_Srepr.npy', mmap_mode = 'r')  # sentence-level (test)
    
    # fMRI data loading
    group_data = np.load('Data/PF_group_neural_data_400Parcel.npy', mmap_mode = 'r')
    
    # start!
    for subj_num in range(31):
        gGLM_modelling(subj_num, group_data, ind_dif_DM, sess_dif_DM, train_DM_Wrepr, test_DM_Wrepr, r'Results/PF/PF_Wctrl_ctxt')
        gGLM_modelling(subj_num, group_data, ind_dif_DM, sess_dif_DM, train_DM_Srepr, test_DM_Srepr, r'Results/PF/PF_Sctrl_ctxt')     

    

