import pandas as pd
import librosa
import soundfile as sf
import os


def check_protol_file(p_file, p_file_orig):

    # read the protcol file
    evalprotcol_df = pd.read_csv(p_file, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])

    print(evalprotcol_df)

    evalprotcol_df_orig = pd.read_csv(p_file_orig, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])

    print(evalprotcol_df_orig)

    for (i, row), (j, row_orig) in zip(evalprotcol_df.iterrows(), evalprotcol_df_orig.iterrows()):

        filename = row["AUDIO_FILE_NAME"]
        filename_orig = row_orig["AUDIO_FILE_NAME"]

        speaker_id = row["Speaker_Id"]
        speaker_id_orig = row_orig["Speaker_Id"]

        system_id = row["SYSTEM_ID"]
        system_id_orig = row_orig["SYSTEM_ID"]

        key = row["KEY"]
        key_orig = row_orig["KEY"]

        str_array = filename.split('_')
        old_filename = str_array[0] + '_' + str_array[1] + '_' + str_array[2]

        # check if corresponding filesnames are correct
        if not filename_orig == old_filename:
            print("XXXXX Filename Wrong XXXXX")
            print(filename_orig)
            print(old_filename)
            break

        # check if corresponfing speaker id is correct
        if not speaker_id_orig == speaker_id:
            print("XXXXX Speaker Id Wrong XXXXX")
            print(speaker_id_orig)
            print(speaker_id)
            break

        
        # check if corresponfing system id is correct
        if not system_id_orig == system_id:
            print("XXXXX System ID Wrong XXXXX")
            print(system_id_orig)
            print(system_id)
            break

        # check if corresponfing Key id is correct
        if not key_orig == key:
            print("XXXXX Key Wrong XXXXX")
            print(key_orig)
            print(key)
            break


def check_audio_valid(path_to_data, cf = False, sample_rate=16000, dst=None):

    audio_files = os.listdir(path_to_data)

    for af in audio_files:

        if cf:
            print(af)
            new_filename = change_filename(af)
            src = path_to_data + af
            dst = path_to_data + new_filename + '.wav'

            if not os.path.exists(dst):
                os.rename(src, dst)  

            af = new_filename + '.wav'

        print(af)
        audio_data, sr = librosa.load(path_to_data + af, mono=True, sr=None)
        # audio_data, sr = sf.read(path_to_data + af)

        print(audio_data.shape)
        print("Before changing the sampling rate {} ".format(sr))

        if not sr == sample_rate:

            print("Changing the sampling rate for file {} ".format(af))

            if dst:
                if not os.path.exists(dst):
                    os.makedirs(dst)
                sf.write(dst + af, audio_data, sample_rate, subtype='PCM_16')
                audio_data, sr = sf.read(dst + af)
            else:
                sf.write(path_to_data + af, audio_data, sample_rate, subtype='PCM_16')
                audio_data, sr = sf.read(path_to_data + af)

            print("After changing the sampling rate {} ".format(sr))


        # if len(audio_data.shape) == 2:
        #     print(af)
        #     print(audio_data.shape)

        librosa.util.valid_audio(audio_data)

        # sf.write(path_to_data + af, audio_data, sr, subtype='PCM_16')


    print("All Audios are valid and converted to mono")



def change_filename(filename, p_file=False):

    # split filename
    filename_ls = filename.split('.')

    if p_file:
        dot_count = 1
    else:
        dot_count = 2

    if len(filename_ls) > dot_count:

        # new_filename = filename_ls[0] + '_' + filename_ls[1]
        new_filename = filename_ls[0]

    else:
        return filename

    return new_filename


def update_protocol_file(p_file):

    # read the protcol file
    evalprotcol_df = pd.read_csv(p_file, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])

    print(evalprotcol_df)

    for index, row in evalprotcol_df.iterrows():

        filename = row["AUDIO_FILE_NAME"]

        new_filename = change_filename(filename, p_file=True)

        if not new_filename == filename:
            
            evalprotcol_df.at[index, "AUDIO_FILE_NAME"] = new_filename

    
    print(evalprotcol_df)

    evalprotcol_df.to_csv(p_file, sep=" ", index=False, header=False)


# path to the database
# data_dir = '/home/alhashim/Data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac/'  
# data_dir = '/data/Data/AsvSpoofData_2019_RT_0_6/'

data_name = 'AsvSpoofData_2019_WN_20_20_5/'
data_dir = '/data/data/' + data_name

dst_dir = '/data/Data/Noise_Addition/' + data_name

# path to protocol file
protocol_file = '/data/Data/AsvSpoofData_2019_protocols/low_pass_filt_7000_protocol.txt'

orig_protocol_file = '../ASVspoof2019.LA.cm.eval.trl.txt'

# check_protol_file(protocol_file, orig_protocol_file)

# check_audio_valid(data_dir, cf=False, dst=dst_dir)

update_protocol_file(protocol_file)

# # read the protcol file
# evalprotcol_df = pd.read_csv(protocol_file, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])

# print(evalprotcol_df)

# evalprotcol_df_orig = pd.read_csv(orig_protocol_file, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])

# print(evalprotcol_df_orig)

# save_processed_audio_dir = 'reverb_0_3'

# # all_files = os.listdir(data_dir)

# filename_ls = []
# loop over the protcol file
# for index, row in evalprotcol_df.iterrows():

#     filename = row["AUDIO_FILE_NAME"]

#     print(filename)

#     # if not filename == 'LA_E_2834763_RT_0_3':

#     dst = data_dir + filename + '.wav'
#     print(dst)

#     if not os.path.exists(dst):

#         # if not filename == 'LA_E_8070617_RT_0_6':

#         str_array = filename.split('_')

#         old_filename = str_array[0] + '_' + str_array[1] + '_' + str_array[2] + '_' + str_array[3] + '_' + str_array[4] + '.' + str_array[5]

#         print(old_filename)
        
#         src = data_dir + old_filename + '.wav'
        

#         print(src)

#         os.rename(src, dst)

    # break

    # # path to the file
    # full_file = data_dir + filename + '.flac'

    # # read audio file
    # audio_data, sr = librosa.load(full_file, sr=None)

    # # pass this audio data to your postprocessing function, for example reverb_3 applies reverberation of T60=0.3 to the audio
    # # reverb_3_audio = reverb_3(audio_data) # just an example

    # # write the new file to the specified location
    # new_filename = 'L' + filename
    # new_filename = filename.split('.')[0] + '_' + filename.split('.')[1]
    # sf.write(new_filename + '.flac', reverb_3_audio, sr, format='flac', subtype='PCM_16')

    # filename_ls.append(new_filename)

# change filenames in the pandas dataframe
# evalprotcol_df["AUDIO_FILE_NAME"] = filename_ls

# save pandas dataframe as txt file
# evalprotcol_df.to_csv(protocol_file, sep=" ", index=False, header=False)






    





    

