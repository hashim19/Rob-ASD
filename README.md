# Rob-ASD

This repository contains code for the paper titled "Is Audio Spoof Detection Robust to Laundering Attacks?". Seven state-of-the-art (SOTA) Audio Spoof Detection systems are evaluated against laundering attacks. These systems are, 
- CQCC-GMM (https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-CQCC-GMM/python)
- LFCC-GMM (https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-LFCC-GMM/python)
- LFCC-LCNN (https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-LFCC-LCNN)
- OC-Softmax (https://github.com/hashim19/AIR-ASVspoof)
- RawNet2 (https://github.com/eurecom-asp/rawnet2-antispoofing)
- RawGAT-ST (https://github.com/eurecom-asp/RawGAT-ST-antispoofing)
- AASIST (https://github.com/clovaai/aasist)

## Folder Structure

The repository is structured as follows:

| Folder | Description                                       |
|--------|---------------------------------------------------|
|__Audio Spoof Detection Systems__|
| `/ASD_ML/` | Contains code for CQCC-GMM and LFCC-GMM systems|
| `/LFCC-LCNN/` | Contains code for LFCC-LCNN system|
| `/AIR-ASVspoof/` | Contains code for OC-Softmax system|
| `/RawNet2/` | Contains code for RawNet2 system|
| `/RawGAT-ST-antispoofing/` | Contains code for RawGAT system|
| `/aasist/` | Contains code for AASIST system|
|__Evaluation__|
| `/Score_Files/` | Contains already evaluated score files|

