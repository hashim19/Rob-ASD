import numpy as np
from numpy import log, exp, infty, zeros_like, vstack, zeros, errstate, finfo, sqrt, floor, tile, concatenate, arange, meshgrid, ceil, linspace
import math
from scipy.interpolate import interpn
from scipy.special import logsumexp
from scipy.signal import lfilter
from scipy.fft import dct
from os.path import exists
from random import sample
import librosa

from Feature_Library.LFCC_pipeline import lfcc
from CQCC.CQT_toolbox_2013.cqt import cqt
from CQCC.cqcc import cqcc


######################### feature extraction functions for CQCC #######################
def cqccDeltas(x, hlen=2):
    win = list(range(hlen, -hlen-1, -1))
    norm = 2*(arange(1, hlen+1)**2).sum()
    xx_1 = tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx) / norm
    return D[:, hlen*2:]

# @jit(nopython=True)
def extract_cqcc(sig, fs, fmin=62.5, fmax=4000, B=12, cf=19, d=16):
    # cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    kl = B * math.log(1 + 1 / d, 2)
    gamma = 228.7 * (2 ** (1 / B) - 2 ** (-1 / B))
    eps = 2.2204e-16
    scoeff = 1

    new_fs = 1/(fmin*(2**(kl/B)-1))
    ratio = 9.562 / new_fs

    Xcq = cqt(sig[:, None], B, fs, fmin, fmax, 'rasterize', 'full', 'gamma', gamma)
    absCQT = abs(Xcq['c'])

    TimeVec = arange(1, absCQT.shape[1] + 1).reshape(1, -1)
    TimeVec = TimeVec * Xcq['xlen'] / absCQT.shape[1] / fs

    FreqVec = arange(0, absCQT.shape[0]).reshape(1, -1)
    FreqVec = fmin * (2 ** (FreqVec / B))

    LogP_absCQT = log(absCQT ** 2 + eps)

    n_samples = int(ceil(LogP_absCQT.shape[0] * ratio))
    Ures_FreqVec = linspace(FreqVec.min(), FreqVec.max(), n_samples)

    # print(TimeVec[0, :])
    # print(Ures_FreqVec)
    xi, yi = meshgrid(TimeVec[0, :], Ures_FreqVec)
    Ures_LogP_absCQT = interpn(points=(TimeVec[0, :], FreqVec[0, :]), values=LogP_absCQT.T, xi=(xi, yi), method='splinef2d')

    CQcepstrum = dct(Ures_LogP_absCQT, type=2, axis=0, norm='ortho')
    CQcepstrum_temp = CQcepstrum[scoeff - 1:cf + 1, :]
    deltas = cqccDeltas(CQcepstrum_temp.T).T

    CQcc = concatenate([CQcepstrum_temp, deltas, cqccDeltas(deltas.T).T], axis=0)

    return CQcc.T


######################### feature extraction functions for LFCC #######################

def lfccDeltas(x, width=3):
    hlen = int(np.floor(width/2))
    win = list(range(hlen, -hlen-1, -1))
    xx_1 = np.tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = np.tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = np.concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx)
    return D[:, hlen*2:]


def extract_lfcc(audio_data, sr, num_ceps=20, order_deltas=2, no_Filters=70):

    lfccs = lfcc(sig=audio_data,
                 fs=sr,
                 num_ceps=num_ceps,
                 nfilts=no_Filters,
                 low_freq=0,
                 high_freq=4000).T
    
    if order_deltas > 0:
        feats = list()
        feats.append(lfccs)
        for d in range(order_deltas):
            feats.append(lfccDeltas(feats[-1]))
        lfccs = np.vstack(feats)

    if num_ceps == 1 and order_deltas == 0:
        print("reducing lfcc size")
        lfccs_1d= lfccs.ravel()
        if lfccs_1d.shape[0] < 2003:
            lfccs_1d_mod = np.pad(lfccs_1d, (0, 2003 - lfccs_1d.shape[0]), 'mean')
            return lfccs_1d_mod
        else:
            return lfccs_1d[:2003]
        # return lfccs.ravel()

    else:
        return lfccs.T
        # return lfccs[:, :2003].T


def extract_mfcc(audio_data, sr):

    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr)

    return mfcc.T


