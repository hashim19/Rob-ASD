import numpy as np
import pandas as pd

import os
import shutil
import librosa
import soundfile as sf


# db_folder = '/data/Data/AsvSpoofData_2019/train/LA'
# data_dir = os.path.join(db_folder, 'ASVspoof2019_LA_train/flac/')

db_folder = '/data/Data/ASVSpoofData_2019_train_laundered'
data_dir = os.path.join(db_folder, 'ASVSpoofData_2019_train_10_percent_laundered/wav')

# protocol_path = os.path.join(db_folder, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.train.trn.txt')
protocol_path = os.path.join(db_folder, 'protocols', 'ASVSpoofData_2019_train_10_percent_laundered_protocol.txt')

out_folder = os.path.join(db_folder, 'ASVSpoofData_2019_train_10_percent_laundered')

out_flac_folder = os.path.join(out_folder, 'flac')

if not os.path.exists(out_flac_folder):
    os.makedirs(out_flac_folder)

df = pd.read_csv(protocol_path, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_For_LA", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])

print(df.head())

# randomly select 10% rows from the dataframe
# df = df.sample(frac=0.1)

# print(df.head())
# print(df.shape)

for index, row in df.iterrows():

    filename = row["AUDIO_FILE_NAME"]
    laundering_type = row["Laundering_Type"]

    if laundering_type == 'Recompression':
        filename_ls = filename.split('_')
        new_filename = filename_ls[0] + '_' + filename_ls[1] + '_' + filename_ls[2] + '_' + filename_ls[-1]
        print(new_filename)
        data, samplerate = librosa.load(data_dir + '/' + new_filename + '.wav', sr=None)
    else:
        data, samplerate = librosa.load(data_dir + '/' + filename + '.wav', sr=None)

    print(data.shape)
    print(samplerate)

    print(filename)
    print("Writing to flac format")
    sf.write(out_flac_folder + '/' + filename + '.flac', data, samplerate)
    

#      filename = row["AUDIO_FILE_NAME"]

#      src_fullfilename = os.path.join(data_dir, filename + '.flac')

#      dst_fullfilename = os.path.join(out_flac_folder, filename + '.flac')

#      shutil.copyfile(src_fullfilename, dst_fullfilename)


# df.to_csv(os.path.join(out_folder, 'ASVspoof2019.LA.cm.train.trn.txt'), header=False, index=False, sep=" ")