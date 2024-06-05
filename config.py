
################ set these parameters ###############

# database type
db_type = 'asvspoof_eval_laundered' # asvspoof_eval_laundered or in_the_wild or asvspoof_train or asvspoof_train_laundered

# path to the database
# db_folder = '/data/Data/ASVSpoofData_2019_train_laundered'
# db_folder = '/data/Data/ds_wild/'
db_folder = '/data/Data/ASVSpoofLaunderedDatabase/ASVSpoofLaunderedDatabase'

# only used for training
data_names = ['ASVspoof2019_LA_train', 'ASVspoof2019_LA_dev', 'ASVSpoofData_2019_train_10_percent_laundered']
# data_names = ['flac']

# laundering type and laundering parameter, 
# look at the readme file of the database for laundering parameters for each laundering type

laundering_type = 'Noise_Addition' # Recompression, Reverberation, Filtering, Resampling, train_laundered for laundered training, wild for in the wild
laundering_param = 'babble_10'         # random for laundered traning, wild for in the wild

# laundering_type = 'train_laundered' # Recompression, Reverberation, Filtering, Resampling, train_laundered for laundered training, wild for in the wild
# laundering_param = 'random'         # random for laundered traning, wild for in the wild

# feature output directory
feat_dir = '/data/Features/'

# type of feature to compute, one of cqcc, lfcc, or mfcc
feature_type = 'cqcc'
feature_format = 'h5f' # h5f or pkl

audio_ext = '.flac'
# data_types = ['train', 'dev', 'train']
data_types = ['eval']
data_labels = ['bonafide', 'spoof']

# model parameters
n_comp = 512

# score dir
score_dir = '../Score_Files_laundered_train'

############# no need to set this parameter unless the name of the database is changed #############
# if db_type == 'in_the_wild':
# protocol_filenames = ['wild_meta.txt']
# protocol_filenames = ['ASVspoof2019.LA.cm.train.trn.txt', 'ASVspoof2019.LA.cm.dev.trl.txt', 'ASVSpoofData_2019_train_10_percent_laundered_protocol.txt']
# else:
protocol_filenames = ['ASVspoofLauneredDatabase_' + laundering_type + '.txt']