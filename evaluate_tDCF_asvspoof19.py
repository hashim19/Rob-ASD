import os
import numpy as np
import pandas as pd
import eval_metrics as em
import matplotlib.pyplot as plt


def gen_score_file(score_file, protocl_file, out_dir='out_dir'):

    # read protocol file using pandas 
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

    Score_file_has_labels = True
    score_file_dir = 'Score_Files/'

    # protocl_file = score_file_dir + 'Protocol_ASV_RT_0_9.txt'
    # protocl_file = score_file_dir + 'street_0_0_5_protocol.txt'
    # protocl_file = score_file_dir + 'recompression_protocol_320k.txt'
    # protocl_file = score_file_dir + 'resample_44100.txt'
    protocl_file = score_file_dir + 'low_pass_filt_7000_protocol.txt'

    # score_file = score_file_dir + 'log_eval_ASVspoof2019_LA_eval_score.txt'
    # score_file = score_file_dir + 'scores-cqcc-gmm-512-filtering-7000-asvspoof19-LA.txt'
    # score_file = score_file_dir + 'LFCC_LCNN_eval_low_pass_filt_7000_score.txt'
    # score_file = score_file_dir + 'RawNet2_low_pass_filt_7000_eval_CM_scores.txt'

    gen_score_file = score_file_dir + 'scores-cqcc-gmm-512-filtering-7000-asvspoof19-LA-labels.txt'
    # score_file = 'score.txt'

    if Score_file_has_labels:

        compute_equal_error_rate(gen_score_file)

    else:

        gen_score_file(score_file, protocl_file, out_dir=score_file_dir)




