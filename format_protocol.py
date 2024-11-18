import pandas as pd
import librosa
import soundfile as sf
import os
from pathlib import Path
import soundfile as sf
import math


db_path = '/data/ASVSpoof5/protocols'
protocol_filename = 'ASVSpoof5_train_laundered_protocol_10_percent.txt'

protcol_file = os.path.join(db_path, protocol_filename)

protocol_df = pd.read_csv(protcol_file, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "Gender", "Not_Used_For_LA", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])

print(protocol_df)

############### for sampling 10% from the laundered dataset ##################

# laundered_df = protocol_df.dropna()

# print(laundered_df)

# non_laundered_df = protocol_df[protocol_df.isnull().any(1)]

# print(non_laundered_df)
# print(non_laundered_df.shape)

# # randomly sample 50% of the laundered dataframe
# laundered_df_10_percent = laundered_df.sample(frac=0.5)

# print(laundered_df_10_percent)

# # merge non laundered and laundered 10 percent dataframe

# df_10_percent = pd.concat([non_laundered_df, laundered_df_10_percent])

# print(df_10_percent)

# # save the dataframe
# df_10_percent.to_csv(os.path.join('/data/', protocol_filename.split('.')[0] + '_' '10_percent.txt'), header=False, index=False, sep=" ")


################# For sampling 10% from the cascaded laundered dataset ##############

# get non_laundered dataframe
non_laundered_df = protocol_df[protocol_df.isnull().any(1)]

print(non_laundered_df)

# read cascaded protocol file
cascaded_protocol_filename = 'ASVspoof5_train_metadata_20_cascading.txt'

cascaded_protcol_file = os.path.join(db_path, cascaded_protocol_filename)

cascaded_protocol_df = pd.read_csv(cascaded_protcol_file, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "Gender", "Not_Used_For_LA", "SYSTEM_ID", "KEY", "Cascading", "Laundering_Type", "Laundering_Param"])

print(cascaded_protocol_df)


# randomly sample 50% of the cascaded laundered dataframe
cascaded_laundered_df_10_percent = cascaded_protocol_df.sample(frac=0.5)

print(cascaded_laundered_df_10_percent)

# concatenate non_laundered dataframe with cascaded laundered dataframe

laundered_df_10_percent = protocol_df

laundered_df_10_percent.insert(loc=6, column="Cascading", value='No')

print(laundered_df_10_percent)

laundered_cascaded_10_percent_df = pd.concat([laundered_df_10_percent, cascaded_laundered_df_10_percent])

print(laundered_cascaded_10_percent_df)

bonafide_df = laundered_df_10_percent[laundered_df_10_percent['KEY'] == 'bonafide']
spoof_df = laundered_df_10_percent[laundered_df_10_percent['KEY'] == 'spoof']

print(bonafide_df)
print(spoof_df)

# laundered_cascaded_10_percent_df.to_csv(os.path.join('/data/', 'ASVSpoof5_train_laundered_cascaded_10_percent.txt'), header=False, index=False, sep=" ")