from gmm_asvspoof import scoring
import pandas
import numpy as np
import os

# configs
features = 'cqcc'

# configs - GMM parameters
ncomp = 512

processing = '-filtering-7000'
# processing = '-RT-0-3'

# score file to write
scores_file = 'scores-' + features + '-gmm-' + str(ncomp) + processing + '-asvspoof19-LA.txt' 

# bona_path = 'gmm_' + str(ncomp) + '_LA_cqcc' + '_bonafide'
# spoof_path = 'gmm_' + str(ncomp) + '_LA_cqcc' + '_spoof'

model_dir = 'gmm_512_LA_' + features
bona_path = os.path.join(model_dir, 'bonafide', 'gmm_final.pkl')
spoof_path = os.path.join(model_dir, 'spoof', 'gmm_final.pkl')

dict_file = dict()
dict_file['bona'] = bona_path
dict_file['spoof'] = spoof_path

db_folder = '/data/Data/'  # put your database root path here

# laundering_type = 'Reverberation/'
# laundering = 'AsvSpoofData_2019_RT_0_3/'
# protocol_pth = 'Protocol_ASV_RT_0_3.txt'

# laundering_type = 'Noise_Addition/'
# laundering = 'AsvSpoofData_2019_WN_20_20_5/'
# protocol_pth = 'white_20_20_5_protocol.txt'

# laundering_type = 'Recompression/'
# laundering = 'recompression_16k'
# protocol_pth = 'recompression_protocol_16k.txt'

# laundering_type = 'Resampling/'
# laundering = 'resample_11025/'
# protocol_pth = 'resample_11025.txt'

laundering_type = 'Filtering/'
laundering = 'low_pass_filt_7000/'
protocol_pth = 'low_pass_filt_7000_protocol.txt'

# laundering_type = 'Transcoding/'
# laundering = 'Asvspoof19_40_audio_facebook/'
# protocol_pth = 'Asvspoof19_40_protocol.csv'

eval_folder = db_folder + laundering_type + laundering
eval_ndx = db_folder + 'AsvSpoofData_2019_protocols/' + protocol_pth

# feat_dir = '/data/Features/AsvSpoofData_2019_RT_0_9/'
feat_dir = os.path.join('/data/Features/', laundering_type, laundering)

audio_ext = '.wav'

# run on ASVspoof 2021 evaluation set
scoring(scores_file=scores_file, dict_file=dict_file, features=features,
        eval_ndx=eval_ndx, eval_folder=eval_folder, audio_ext=audio_ext,
        feat_dir=feat_dir, features_cached=True)