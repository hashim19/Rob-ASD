# from gmm import train_gmm
# from gmm_sklearn import train_gmm
# from gmm_torch import train_gmm
from gmm_asvspoof import train_gmm
import os
import pickle
import pandas


# configs - feature extraction e.g., LFCC or CQCC
features = 'lfcc'

# configs - GMM parameters
ncomp = 512

# GMM pickle file
dict_file = 'gmm_' + str(ncomp) + '_LA_' + features

# configs - train & dev data - if you change these datasets
db_folder = '/home/alhashim/Data/AsvSpoofData_2019/train/'
train_folders = [db_folder + 'LA/ASVspoof2019_LA_train/flac/']  # [db_folder + 'LA/ASVspoof2019_LA_train/flac/', db_folder + 'LA/ASVspoof2019_LA_dev/flac/']
train_keys = [db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt']  # [db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trn.txt']

audio_ext = '.flac'

data_labels = ['bonafide', 'spoof']

feat_dir = './features_out'

# train bona fide & spoof GMMs
# gmm_bona = train_gmm(data_label=data_labels[0], features=features,
#                         train_keys=train_keys, train_folders=train_folders, audio_ext=audio_ext,
#                         dict_file=dict_file, ncomp=ncomp, feat_dir=feat_dir,
#                         init_only=False)
gmm_spoof = train_gmm(data_label=data_labels[1], features=features,
                        train_keys=train_keys, train_folders=train_folders, audio_ext=audio_ext,
                        dict_file=dict_file, ncomp=ncomp, feat_dir=feat_dir,
                        init_only=False)

