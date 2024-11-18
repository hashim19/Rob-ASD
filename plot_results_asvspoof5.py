import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from astropy.table import Table

# configurations
plot_type = 'comparison_bar_plot' # laundering_attack_plot or db_plot

save_fig = 'False'

plot_dir = './ASVspoof5_results/Figures'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)

############## read the results files using astropy tab #############
mindcf_file = './ASVspoof5_results/min_dcf.tex'
actdcf_file = './ASVspoof5_results/act_dcf.tex'
cllr_file = './ASVspoof5_results/cllr.tex'
eer_file = './ASVspoof5_results/eer.tex'

mindcf_tab = Table.read(mindcf_file)
actdcf_tab = Table.read(actdcf_file)
cllr_tab = Table.read(cllr_file)
eer_tab = Table.read(eer_file)

print(mindcf_tab)
print(actdcf_tab)
print(cllr_tab)
print(eer_tab)

################# Convert the astropy tables to pandas dataframes ##########

mindcf_df = mindcf_tab.to_pandas()
actdcf_df = actdcf_tab.to_pandas()
cllr_df = cllr_tab.to_pandas()
eer_df = eer_tab.to_pandas()

mindcf_df = mindcf_df.set_index('col0').rename_axis('Spoofing Attacks')
actdcf_df = actdcf_df.set_index('col0').rename_axis('Spoofing Attacks')
cllr_df = cllr_df.set_index('col0').rename_axis('Spoofing Attacks')
eer_df = eer_df.set_index('col0').rename_axis('Spoofing Attacks')

print(mindcf_df)
print(actdcf_df)
print(cllr_df)
print(eer_df)

pooled_attack_mindcf_df = mindcf_df['pooled']


ax = pooled_attack_mindcf_df.plot(kind='bar', x='Spoofing Attacks', xlabel='', ylabel='min DCF', rot=15)
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper right')

for c in ax.containers:
    ax.bar_label(c, fmt='%.2f', label_type='edge', fontsize=8)

filename = '_'.join(("bar_plot", 'pooled_attack_mindcf'))
fullfile = os.path.join(plot_dir, filename)
if save_fig:
    plt.savefig(fullfile + '.png', bbox_inches='tight')

# print(pooled_attack_df)