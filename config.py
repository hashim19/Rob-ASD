
################ set these parameters ###############

# database type
db_type = 'in_the_wild' # asvspoof or in_the_wild

# path to the database
# db_folder = '/data/Data/ASVSpoofLaunderedDatabase/ASVSpoofLaunderedDatabase/'
db_folder = '/data/Data/ds_wild/'

# laundering type and laundering parameter, 
# look at the readme file of the database for laundering parameters for each laundering type

laundering_type = 'wild' # Recompression, Reverberation, Filtering, Resampling
laundering_param = 'wild'     

# feature output directory
feat_dir = '/data/Features/'

# type of feature to compute, one of cqcc, lfcc, or mfcc
feature_type = 'cqcc'

audio_ext = '.wav'
data_types = ['eval']
data_labels = ['bonafide', 'spoof']

# model parameters
n_comp = 512

# score dir
score_dir = '../Score_Files'

############# no need to set this parameter unless the name of the database is changed #############
# if db_type == 'in_the_wild':
protocol_filename = 'wild_meta.txt'
# else:
# protocol_filename = 'ASVspoofLauneredDatabase_' + laundering_type + '.txt'