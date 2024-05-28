import os
import numpy as np
import pandas as pd
import argparse
import eval_metrics as em
import matplotlib.pyplot as plt

import config as config


def gen_score_file(score_file, protocl_file, out_dir='out_dir'):

    # read protocol file using pandas 
    if config.db_type == 'in_the_wild':
        evalprotcol = pd.read_csv(protocl_file, sep=',', names=["AUDIO_FILE_NAME", "Speaker_Id", "KEY"])
    elif config.db_type == 'asvspoof':      
        evalprotcol = pd.read_csv(protocl_file, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])
    # evalprotcol = pd.read_csv(protocl_file, sep=",", names=["Speaker_Id", "AUDIO_FILE_NAME", "SYSTEM_ID", "KEY"])

    print(evalprotcol)

    # read score file
    evalscores = pd.read_csv(score_file, sep=" ", names=["AUDIO_FILE_NAME", "Scores"])
    
    # print(evalscores)

    merged_df = pd.merge(evalprotcol, evalscores, on='AUDIO_FILE_NAME')

    # print(merged_df)

    score_df = merged_df[['AUDIO_FILE_NAME', 'KEY', 'Scores']]

    print(score_df)

    out_file = score_file.split('/')[-1].split('.')[0]

    out_file = out_file + '-labels.txt'

    score_df.to_csv(out_dir + out_file, sep=" ", header=None, index=False)

def compute_equal_error_rate(cm_score_file):

    print(cm_score_file)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]

    cm_keys = cm_data[:, 1]
    cm_scores = cm_data[:, 2].astype(float)

    # cm_sources = cm_data[:, 1]
    # cm_keys = cm_data[:, 2]
    # cm_scores = cm_data[:, 3].astype(float)

    # print(cm_utt_id)
    # print(cm_keys)
    # print(cm_scores)

    # cm_data = pd.read_csv(cm_score_file)

    # print(cm_data)

    # cm_utt_id = cm_data['AUDIO_FILE_NAME'].to_list()
    # cm_keys = cm_data['KEY'].to_list()
    # cm_scores = cm_data['Scores'].to_list()

    # other_cm_scores = -cm_scores

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    # other_eer_cm = em.compute_eer(other_cm_scores[cm_scores > 0], other_cm_scores[cm_scores <= 0])[0]

    # print(other_eer_cm)

    print('\nCM SYSTEM')
    # print('   EER            = {:8.5f} % (Equal error rate for countermeasure)'.format(min(eer_cm, other_eer_cm) * 100))
    print('   EER            = {:8.5f} % (Equal error rate for countermeasure)'.format(eer_cm * 100))

    # return min(eer_cm, other_eer_cm)
    return eer_cm



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation script')

    parser.add_argument('--score_file_has_keys', action='store_true', help='if score file has keys')
    parser.add_argument('--score_file_dir', type=str, default='Score_Files/', help='Score File directory')
    parser.add_argument('--protocol_filename', type=str, default='ASVspoof2019.LA.cm.eval.trl.txt', help='Path to the protocol file')
    parser.add_argument('--score_filename', type=str, default='scores-lfcc-gmm-512-asvspoof19-LA.txt', help='Path to the score file')

    args = parser.parse_args()

    # protocol_file_path = args.score_file_dir + args.protocol_filename
    protocol_file_path = os.path.join(config.db_folder, 'protocols', args.protocol_filename)
    score_file_path = args.score_file_dir + args.score_filename

    if args.score_file_has_keys:
        compute_equal_error_rate(score_file_path)
    
    else:
        gen_score_file(score_file_path, protocol_file_path, out_dir=args.score_file_dir)
        compute_equal_error_rate(score_file_path.split('.')[0] + '-labels.txt')




