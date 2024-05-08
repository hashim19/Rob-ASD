
################ set these parameters ###############

# path to the database
db_folder = '/data/Data/ASVSpoofLaunderedDatabase/ASVspoofLauneredDatabase/'

# laundering type and laundering parameter, 
# look at the readme file of the database for laundering parameters for each laundering type

laundering_type = 'Noise_Addition' # Recompression, Reverberation, Filtering, Resampling
laundering_param = 'babble_10'     

# feature output directory
feat_dir = '/data/Features_2/'

# type of feature to compute, one of cqcc, lfcc, or mfcc
feature_type = 'lfcc'

audio_ext = '.flac'
data_types = ['eval']
data_labels = ['bonafide', 'spoof']

n_comp = 512

############# no need to set this parameter unless the name of the database is changed #############
protocol_filename = 'ASVspoofLauneredDatabase_' + laundering_type + '.txt'