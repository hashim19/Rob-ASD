from gmm_asvspoof import scoring
import pandas
import numpy as np
import os
import sys
sys.path.append("../")

import config as config
import pandas as pd

# configs
features = config.feature_type

# configs - GMM parameters
ncomp = config.n_comp

db_folder = config.db_folder  # put your database root path here
db_type = config.db_type

laundering_type = config.laundering_type
laundering_param = config.laundering_param
protocol_pth = config.protocol_filenames[0]

score_dir = config.score_dir

if not os.path.exists(score_dir):
        os.makedirs(score_dir)

# score file to write
scores_file = os.path.join(score_dir, features + '-gmm-' + str(ncomp) + '-' + laundering_param + '-' + config.db_type + '-scores' + '.txt')

print(scores_file)

# bona_path = 'gmm_' + str(ncomp) + '_LA_cqcc' + '_bonafide'
# spoof_path = 'gmm_' + str(ncomp) + '_LA_cqcc' + '_spoof'

# model_dir = 'gmm_' + str(ncomp) + '_LA_' + features + '_' + 'train_laundered'
model_dir = 'gmm_' + str(ncomp) + '_LA_' + features
bona_path = os.path.join(model_dir, 'bonafide', 'gmm_final.pkl')
spoof_path = os.path.join(model_dir, 'spoof', 'gmm_final.pkl')

dict_file = dict()
dict_file['bona'] = bona_path
dict_file['spoof'] = spoof_path

# laundering_type = 'Transcoding/'
# laundering = 'Asvspoof19_40_audio_facebook/'
# protocol_pth = 'Asvspoof19_40_protocol.csv'

if db_type == 'in_the_wild':
        eval_folder = os.path.join(db_folder, 'release_in_the_wild')
elif db_type == 'asvspoof_eval_laundered':
        eval_folder = os.path.join(db_folder, 'flac')

elif db_type == 'asvspoof_eval':
        eval_folder = os.path.join(db_folder, 'flac')

eval_ndx = os.path.join(db_folder, 'protocols', protocol_pth)

feat_dir = os.path.join(config.feat_dir, laundering_type, laundering_param)

audio_ext = config.audio_ext

if db_type == 'in_the_wild':
        df = pd.read_csv(eval_ndx, sep=',', names=["AUDIO_FILE_NAME", "Speaker_Id", "KEY"])
        files = df["AUDIO_FILE_NAME"].values
        print(df)
        print(len(files))

elif db_type == 'asvspoof_eval_laundered':
        df = pd.read_csv(eval_ndx, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])
        df = df[df["Laundering_Param"] == laundering_param]
        files = df["AUDIO_FILE_NAME"].values
        print(df)
        print(len(files))

elif db_type == 'asvspoof_eval':
        print(eval_ndx)
        df = pd.read_csv(eval_ndx, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_For_LA", "SYSTEM_ID", "KEY"])
        files = df["AUDIO_FILE_NAME"].values
        print(df)
        print(len(files))

# run on ASVspoof 2021 evaluation set
test_scores = scoring(scores_file=scores_file, dict_file=dict_file, features=features,
        eval_file_list=files, eval_folder=eval_folder, audio_ext=audio_ext,
        feat_dir=feat_dir, features_cached=True)