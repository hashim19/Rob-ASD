import pandas as pd
import librosa
import soundfile as sf
import os
from pathlib import Path
import soundfile as sf

Data_path = '/data/Data/'

laundering_type = 'Filtering'

# laundering_parameters = ['babble_0_0_5', 'babble_10_10_5', 'babble_20_20_5', 'cafe_0_0_5', 'cafe_10_10_5', 'cafe_20_20_5', 'street_0_0_5', 'street_10_10_5', 
#                         'street_20_20_5', 'volvo_0_0_5', 'volvo_10_10_5', 'volvo_20_20_5', 'white_0_0_5', 'white_10_10_5', 'white_20_20_5']

# laundering_parameters = ['RT_0_3', 'RT_0_6', 'RT_0_9']

# laundering_parameters = ['recompression_128k', 'recompression_16k', 'recompression_196k', 'recompression_256k', 'recompression_320k', 'recompression_64k']

# laundering_parameters = ['resample_11025', 'resample_22050', 'resample_44100', 'resample_8000']

laundering_parameters = ['lpf_7000']

# laundering_type = 'Filtering/'
# laundering = 'low_pass_filt_7000/'
# protocol_pth = 'low_pass_filt_7000_protocol.txt'

out_data_dir_name = 'ASVspoofLauneredDatabase'

# create the output directory
out_dir = os.path.join(Data_path, out_data_dir_name)
# flac_out_dir = os.path.join(out_dir, 'ASVspoof2019_LA_eval_' + laundering_type, 'flac')
flac_out_dir = os.path.join(out_dir, 'flac')

if not os.path.exists(flac_out_dir):
    os.makedirs(flac_out_dir)

# read original protocol file
protocol_file_orig = 'ASVspoof2019.LA.cm.eval.trl.txt'

evalprotcol_df_orig = pd.read_csv(protocol_file_orig, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])

out_protocol_dict = {"Speaker_Id": [], "AUDIO_FILE_NAME": [], "SYSTEM_ID": [], "KEY": [], "Laundering_Type": [], "Laundering_Param": []}

for lp in laundering_parameters:

    # build data directory path
    if laundering_type == 'Noise_Addition' or laundering_type == 'Reverberation':

        data_dir = os.path.join(Data_path, laundering_type, 'AsvSpoofData_2019_' + lp)
    
    else:

        data_dir = os.path.join(Data_path, laundering_type, lp)


    # wav_files = list(Path(data_dir).glob("*.wav"))

    # build protocol path
    protocol_file_lp = os.path.join(Data_path, 'AsvSpoofData_2019_protocols', lp + '_protocol.txt')

    evalprotcol_df = pd.read_csv(protocol_file_lp, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])

    for (i, row), (j, row_orig) in zip(evalprotcol_df.iterrows(), evalprotcol_df_orig.iterrows()):

        filename = row["AUDIO_FILE_NAME"]
        filename_orig = row_orig["AUDIO_FILE_NAME"]

        speaker_id = row["Speaker_Id"]
        speaker_id_orig = row_orig["Speaker_Id"]

        system_id = row["SYSTEM_ID"]
        system_id_orig = row_orig["SYSTEM_ID"]

        key = row["KEY"]
        key_orig = row_orig["KEY"]

        # save the laundered audio file at the desired location with a desired name in a desired format
        if laundering_type == 'Noise_Addition':

            lp_ls = lp.split('_')
            lp= lp_ls[0] + '_' + lp_ls[1]

        af_out_laundered = filename_orig + '_' + lp
        af_out_laundered_fullfile = os.path.join(flac_out_dir, af_out_laundered + '.flac')
        
        if not os.path.exists(af_out_laundered_fullfile):

            # read and write laundered audio file
            af_in_laundered = os.path.join(data_dir, filename + '.wav')

            audio_data, sr = librosa.load(af_in_laundered, mono=True, sr=None)

            print("writing to file {}".format(af_out_laundered))

            sf.write(af_out_laundered_fullfile, audio_data, sr)
        
        else:
            print("file {} already written".format(af_out_laundered))

        out_protocol_dict['AUDIO_FILE_NAME'].append(af_out_laundered)  
        out_protocol_dict['Speaker_Id'].append(speaker_id)  
        out_protocol_dict['SYSTEM_ID'].append(system_id)
        out_protocol_dict['KEY'].append(key)
        out_protocol_dict['Laundering_Type'].append(laundering_type)
        out_protocol_dict['Laundering_Param'].append(lp)


out_protocol_df = pd.DataFrame(out_protocol_dict)

out_protocol_file = out_dir + '/' + out_data_dir_name + '_' + laundering_type + '.txt'

# if os.path.exists(out_protocol_file):
#     base_prot_df = pd.read_csv(out_protocol_file, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])

#     out_protocol_df = pd.concat([base_prot_df, out_protocol_df])

out_protocol_df.to_csv(out_protocol_file, sep=" ", index=False, header=False)

###############  combining multiple dataframes into one  ################
# laundering_types = ['Noise_Addition', 'Reverberation', 'Recompression', 'Resampling', 'Filtering']

# out_data_dir_name = 'ASVspoofLauneredDatabase'

# for lt in laundering_types:

#     protocol_file = os.path.join(Data_path, out_data_dir_name, out_data_dir_name + '_' + lt + '.txt')

#     base_prot_df = pd.read_csv(protocol_file, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])

#     print(base_prot_df.head)



    