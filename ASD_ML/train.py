# from gmm import train_gmm
# from gmm_sklearn import train_gmm
# from gmm_torch import train_gmm
from gmm_asvspoof import train_gmm
import os
import pickle
import pandas as pd

import sys
sys.path.append("../")

import config as config


# configs - feature extraction e.g., LFCC or CQCC
features = config.feature_type

# configs - GMM parameters
ncomp = config.n_comp

# configs - train & dev data - if you change these datasets
db_folder = config.db_folder
data_names = config.data_names

laundering_type = config.laundering_type
laundering_param = config.laundering_param
protocol_filenames = config.protocol_filenames

# GMM pickle file
dict_file = 'gmm_' + str(ncomp) + '_LA_' + features + '_' + laundering_type

train_folders = [os.path.join(db_folder, dt_name, 'flac') for dt_name in data_names]
train_keys = [os.path.join(db_folder, 'protocols', pf) for pf in protocol_filenames] 

print(train_folders)
print(train_keys)

# train_folders = [db_folder + 'LA/ASVspoof2019_LA_train/flac/']  # [db_folder + 'LA/ASVspoof2019_LA_train/flac/', db_folder + 'LA/ASVspoof2019_LA_dev/flac/']
# train_keys = [db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt']  # [db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trn.txt']

Feat_dir = os.path.join(config.feat_dir, laundering_type, laundering_param)

audio_ext = config.audio_ext
data_types = config.data_types
data_labels = config.data_labels

# train bona fide & spoof GMMs
# gmm_bona = train_gmm(data_label=data_labels[0], features=features,
#                         train_keys=train_keys, train_folders=train_folders, audio_ext=audio_ext,
#                         dict_file=dict_file, ncomp=ncomp, feat_dir=Feat_dir,
#                         init_only=False)
# gmm_spoof = train_gmm(data_label=data_labels[1], features=features,
#                         train_keys=train_keys, train_folders=train_folders, audio_ext=audio_ext,
#                         dict_file=dict_file, ncomp=ncomp, feat_dir=Feat_dir,
#                         init_only=False)

