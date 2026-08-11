
################ set these parameters ###############

# Decide whether to do evaluation or training

# decide the database type
# choose from following.
# { asvspoof_eval_laundered, in_the_wild, asvspoof_train, asvspoof_train_laundered, asvspoof_eval, asvspoof5_train_laundered, 
# asvspoof5_eval_laundered, asvspoof5_train }
# db_type = 'in_the_wild' 
# db_type = 'asvspoof_train_laundered' 
# db_type = 'asvspoof_eval' 
# db_type = 'asvspoof_eval_laundered' 
# db_type = 'asvspoof_train'
db_type = 'FF'

# path to the database
# db_folder = '/data/Data/ASVSpoofData_2019_train_laundered' # for asvspoof_train_laundered
# db_folder = '/data/Data/ds_wild/'  # for in_the_wild
# db_folder = '/data/Data/ASVSpoofLaunderedDatabase/ASVSpoofLaunderedDatabase' # for asvspoof_eval_laundered
db_folder = '/data/Data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval' # for asvspoof_eval
# db_folder = '/data/Data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev' # for asvspoof_dev
# db_folder = '/data/ASVSpoof5/'    # for asvspoof5_train

# only used for training
data_names = ['ASVspoof2019_LA_train/flac', 'ASVspoof2019_LA_dev/flac', 'ASVSpoofData_2019_train_all_laundered/flac_stochastic']
# data_names = ['ASVspoof2019_LA_train/flac', 'ASVspoof2019_LA_dev/flac', 'ASVSpoofData_2019_train_10_percent_laundered/flac']
# data_names = ['flac']
# data_names = ['No_Laundering_train', 'No_Laundering_dev']

# data_names = ['Laundering', 'No_Laundering_dev']        # for asvspoof5
# data_names = ['No_Laundering_progress']        # for asvspoof5

# laundering type and laundering parameter, 
# look at the readme file of the database for laundering parameters for each laundering type

# laundering_type = 'Reverberation' # Recompression, Reverberation, Filtering, Resampling, train_laundered for laundered training, wild for in the wild
# laundering_param = 'RT_0_9'         # random for laundered traning, wild for in the wild

# laundering_type = 'Noise_Addition' # Recompression, Reverberation, Filtering, Resampling, train_laundered for laundered training, wild for in the wild
# laundering_param = 'street_20'         # random for laundered traning, wild for in the wild

# laundering_type = 'Recompression' # Recompression, Reverberation, Filtering, Resampling, train_laundered for laundered training, wild for in the wild
# laundering_param = 'recompression_320k'         # random for laundered traning, wild for in the wild

# laundering_type = 'Filtering' # Recompression, Reverberation, Filtering, Resampling, train_laundered for laundered training, wild for in the wild
# laundering_param = 'lpf_7000'         # random for laundered traning, wild for in the wild

# laundering_type = 'Resampling' # Recompression, Reverberation, Filtering, Resampling, train_laundered for laundered training, wild for in the wild
# laundering_param = 'lpf_7000'         # random for laundered traning, wild for in the wild

# laundering_type = 'train_laundered' # Recompression, Reverberation, Filtering, Resampling, train_laundered for laundered training, wild for in the wild
# laundering_param = 'random'         # random for laundered traning, wild for in the wild

# laundering_type = 'no_laundering' # Recompression, Reverberation, Filtering, Resampling, train_laundered for laundered training, wild for in the wild
# laundering_param = 'no_laundering'         # random for laundered traning, wild for in the wild

# laundering_type = 'wild' # Recompression, Reverberation, Filtering, Resampling, train_laundered for laundered training, wild for in the wild
# laundering_param = 'wild'         # random for laundered traning, wild for in the wild

laundering_type = 'all_laundering_attacks_sampled' # Recompression, Reverberation, Filtering, Resampling, train_laundered for laundered training, wild for in the wild, all for all laundering types
laundering_param = 'all'         # random for laundered traning, wild for in the wild, all for all laundering params

# laundering_type = 'ASVSpoofLD' # Recompression, Reverberation, Filtering, Resampling, train_laundered for laundered training, wild for in the wild, all for all laundering types
# laundering_param = 'sampled'         # random for laundered traning, wild for in the wild, all for all laundering params

# augmentation strategy and variant for training and evaluating augmented training models
Aug_strategy = 'paired'
Aug_type = 'REC_RES'

# feature output directory
feat_dir = '/home/hashim/PhD/Features/'
# feat_dir = '/data/ASVSpoof5_features/'       # for asvspoof5

# type of feature to compute, one of cqcc, lfcc, or mfcc
feature_type = 'lfcc'
feature_format = 'pkl' # h5f or pkl

audio_ext = '.flac'
# data_types = ['train', 'dev']
# data_types = ['train', 'dev', 'train']
data_types = ['eval']

# data_types = ['train', 'dev']                # for asvspoof5
# data_types = ['eval']                # for asvspoof5

data_labels = ['bonafide', 'spoof']

# model parameters
n_comp = 512

# score dir
# score_dir = '../Score_Files_laundered_train'
# score_dir = '../Score_Files_Clean'
# score_dir = '../Score_Files_Single_clean'
# score_dir = '../Score_Files_Paired_clean'
# score_dir = '../Score_Files_cascaded_clean'
# score_dir = '../Score_Files_Single_laundered'
score_dir = '../Score_Files_Paired_laundered'
# score_dir = '../Score_Files_cascaded_laundered'
# score_dir = '../Score_Files_Single_itw'
# score_dir = '../Score_Files_Paired_itw'
# score_dir = '../Score_Files_cascaded_itw'
# score_dir = '../Score_Files_stochastic_clean'
# score_dir = '../Score_Files_stochastic_laundered'
# score_dir = '../Score_Files_stochastic_itw'

# score_dir = '../Score_Files_FakeXpose'

# score_dir = '../Score_Files_asvspoof5_laundered_train'

############# no need to set this parameter unless the name of the database is changed #############
# if db_type == 'in_the_wild':
# protocol_filenames = ['wild_meta.txt']
# protocol_filenames = ['ASVspoof2019.LA.cm.train.trn.txt', 'ASVspoof2019.LA.cm.dev.trl.txt', 'ASVSpoofData_2019_train_10_percent_laundered_protocol.txt']
protocol_filenames = ['ASVspoof2019.LA.cm.train.trn.txt', 'ASVspoof2019.LA.cm.dev.trl.txt', 'stochastic_ASV19.txt']
# else:
# protocol_filenames = ['ASVspoofLauneredDatabase_' + laundering_type + '.txt']

# protocol_filenames = ['ASVspoof2019.LA.cm.eval.trl.txt']
# protocol_filenames = ['ASVspoof2019.LA.cm.dev.trl.txt']

# for asvspoof5
# protocol_filenames = ['ASVspoof5_train_protocol.txt', 'ASVspoof5.dev.metadata.txt']
# protocol_filenames = ['ASVSpoof5_train_laundered_protocol_10_percent.txt', 'ASVspoof5.dev.metadata.txt']       
# protocol_filenames = ['ASVspoof5.track_1.progress.trial.txt']