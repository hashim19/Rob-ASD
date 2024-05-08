from gmm_asvspoof import scoring
import pandas
import numpy as np
import os

import config as config

# configs
features = config.feature_type

# configs - GMM parameters
ncomp = config.n_comp

laundering_param = config.laundering_param
# processing = '-RT-0-3'

# score file to write
scores_file = 'scores-' + features + '-gmm-' + str(ncomp) + '-' + laundering_param + '-asvspoof19-LA.txt' 

# bona_path = 'gmm_' + str(ncomp) + '_LA_cqcc' + '_bonafide'
# spoof_path = 'gmm_' + str(ncomp) + '_LA_cqcc' + '_spoof'

model_dir = 'gmm_' + str(ncomp) + '_LA_' + features
bona_path = os.path.join(model_dir, 'bonafide', 'gmm_final.pkl')
spoof_path = os.path.join(model_dir, 'spoof', 'gmm_final.pkl')

dict_file = dict()
dict_file['bona'] = bona_path
dict_file['spoof'] = spoof_path

db_folder = config.db_folder  # put your database root path here

laundering_type = config.laundering_type
laundering_param = config.laundering_param
protocol_pth = config.protocol_filename

# laundering_type = 'Transcoding/'
# laundering = 'Asvspoof19_40_audio_facebook/'
# protocol_pth = 'Asvspoof19_40_protocol.csv'

eval_folder = os.path.join(db_folder, 'flac')
eval_ndx = os.path.join(db_folder, 'protocols', protocol_pth)

feat_dir = os.path.join(config.feat_dir, laundering_type, laundering_param)

audio_ext = config.audio_ext

# run on ASVspoof 2021 evaluation set
scoring(scores_file=scores_file, dict_file=dict_file, features=features,
        eval_ndx=eval_ndx, eval_folder=eval_folder, audio_ext=audio_ext,
        feat_dir=feat_dir, features_cached=True)