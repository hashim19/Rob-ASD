import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read the results file
res_file = 'Post_Processed_Asvspoof_Results.csv'

res_data = pd.read_csv(res_file)
# res_data = res_data.set_index('Laundering Attack')
print(res_data)

# res_data.plot(y=['CQCC-GMM',  'LFCC-GMM',  'LFCC-LCNN',  'Ocsoftmax',  'Rawnet2',  'RawGat_ST',  'AASIST'], x=['Babble(AVG)', 'Volvo(AVG)', 'WN(AVG)',
#                  'Cafe(AVG)', 'Street(AVG)'])

############### Laundering Attack Plot ############
# res_data_avg = res_data.loc[((res_data['LaA_Parameter']=='avg') & ~(res_data['Laundering Attack']=='RT'))]

# print(res_data_avg)

# ax1=res_data_avg.plot(y=['CQCC-GMM',  'LFCC-GMM',  'LFCC-LCNN',  'Ocsoftmax',  'Rawnet2',  'RawGat_ST',  'AASIST'], x='Laundering Attack', 
#                     figsize=(12,9), marker = 'o')

# ax1.legend(fontsize="x-large", loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
# ax1.set_xlabel('Laundering Attack Type', fontdict={'fontsize':'x-large'})
# ax1.set_ylabel('EER (%)', fontdict={'fontsize':'x-large'})
# ax1.tick_params(axis='both', which='major', labelsize=15)

################# Db plot ###################
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

# plt.show()
# plt.savefig("EER_Plot_Laundering_Attack.png", bbox_inches='tight')
plt.savefig("EER_Plot_DB.png", bbox_inches='tight')

