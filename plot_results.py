import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# configurations
plot_type = 'comparison_bar_plot' # laundering_attack_plot or db_plot

save_fig = 'True'

plot_dir = 'Figures'

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('ytick', labelsize=BIGGER_SIZE)
plt.rc('xtick', labelsize=BIGGER_SIZE)

# read the results file
non_aug_res_file = 'Results_Laundered_Database_training_Non_Augmented.csv'
aug_res_file = 'Results_Laundered_Database_training_augmented.csv'

non_aug_res_data = pd.read_csv(non_aug_res_file)
aug_res_data = pd.read_csv(aug_res_file)
# res_data = res_data.set_index('Laundering Attack')
print(non_aug_res_data)
print(aug_res_data)

# res_data.plot(y=['CQCC-GMM',  'LFCC-GMM',  'LFCC-LCNN',  'Ocsoftmax',  'Rawnet2',  'RawGat_ST',  'AASIST'], x=['Babble(AVG)', 'Volvo(AVG)', 'WN(AVG)',
#                  'Cafe(AVG)', 'Street(AVG)'])

############### Laundering Attack Plot ############
if plot_type == 'laundering_attack_plot':
    res_data_avg = res_data.loc[((res_data['LaA_Parameter']=='avg') & ~(res_data['Laundering Attack']=='RT'))]

    print(res_data_avg)

    ax1=res_data_avg.plot(y=['CQCC-GMM',  'LFCC-GMM',  'LFCC-LCNN',  'Ocsoftmax',  'Rawnet2',  'RawGat_ST',  'AASIST'], x='Laundering Attack', 
                        figsize=(12,9), marker = 'o')

    ax1.legend(fontsize="x-large", loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    ax1.set_xlabel('Laundering Attack Type', fontdict={'fontsize':'x-large'})
    ax1.set_ylabel('EER (%)', fontdict={'fontsize':'x-large'})
    ax1.tick_params(axis='both', which='major', labelsize=15)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if save_fig:
        plt.savefig(plot_dir + '/' + "EER_Plot_Laundering_Attack.png", bbox_inches='tight')

################# Db plot ###################
elif plot_type == 'db_plot':
    res_data_add_noise = res_data.loc[~(res_data['Laundering Attack']=='RT') & ~(res_data['LaA_Parameter']=='avg')]

    print(res_data_add_noise)

    res_data_add_noise = res_data_add_noise.drop(columns=['Laundering Attack'])

    print(res_data_add_noise)

    res_data_db = res_data_add_noise.groupby(['LaA_Parameter']).mean()

    print(res_data_db)

    ax2=res_data_db.plot(y=['CQCC-GMM',  'LFCC-GMM',  'LFCC-LCNN',  'Ocsoftmax',  'Rawnet2',  'RawGat_ST',  'AASIST'], 
                        figsize=(12,9), marker = 'o')

    ax2.legend(fontsize="x-large", loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    ax2.set_xlabel('SNR (db)', fontdict={'fontsize':'x-large'})
    ax2.set_ylabel('EER (%)', fontdict={'fontsize':'x-large'})
    ax2.tick_params(axis='both', which='major', labelsize=15)

    plt.ylim(0, 45)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if save_fig:
        plt.savefig(plot_dir + '/' + "EER_Plot_DB.png", bbox_inches='tight')

elif plot_type == 'comparison_bar_plot':

    # filter only rows with parameter avg
    non_aug_res_data = non_aug_res_data.loc[(non_aug_res_data['LaA_Parameter']=='avg') | (non_aug_res_data['LaA_Parameter']=='7k')]
    aug_res_data = aug_res_data.loc[(aug_res_data['LaA_Parameter']=='avg') | (aug_res_data['LaA_Parameter']=='7k')]

    aug_res_data = aug_res_data.drop(columns=['LaA_Parameter'])
    non_aug_res_data = non_aug_res_data.drop(columns=['LaA_Parameter'])

    print(aug_res_data)
    print(non_aug_res_data)

    non_aug_res_data = non_aug_res_data.set_index('Laundering Attack').T.rename_axis('ASD Systems').reset_index()
    aug_res_data = aug_res_data.set_index('Laundering Attack').T.rename_axis('ASD Systems').reset_index()

    print(aug_res_data)
    print(non_aug_res_data)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    laundering_attacks = ['RT', 'Babble', 'Volvo', 'WN', 'Cafe', 'Street', 'Recompression', 'Resampling', 'Low Pass Filter']

    for ld in laundering_attacks:
        
        df_merged = pd.merge(non_aug_res_data[['ASD Systems', ld]],  aug_res_data[['ASD Systems', ld]], on='ASD Systems')

        df_merged.columns = ['ASD Systems', 'Non Augmented train data', 'Augmented train data']

        ax = df_merged.plot(kind='bar', x='ASD Systems', xlabel='', ylabel='EER (%)', rot=15)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper right')

        for c in ax.containers:
            ax.bar_label(c, fmt='%.2f', label_type='edge', fontsize=8)

        if ld == 'Low Pass Filter':
            filename = '_'.join(("bar_plot", 'LowPassFilter'))
        
        else:
            filename = '_'.join(("bar_plot", ld))

        fullfile = os.path.join(plot_dir, filename)

        if save_fig:
            # plt.savefig(fullfile + '.png', bbox_inches='tight')
            plt.savefig(fullfile + '.eps', bbox_inches='tight', format='eps')

