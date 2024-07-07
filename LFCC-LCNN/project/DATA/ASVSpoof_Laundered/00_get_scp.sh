#!/bin/bash

# Just list the trial names in separate files
# The name convention is based on ASVspoof2019

# database paths
db_folder="/data/Data/ASVSpoofData_2019_train_laundered"
protocol_path="${db_folder}/protocols"

mkdir scp

# train set
train_protocol_path_1="${protocol_path}/ASVspoof2019.LA.cm.train.trn.txt "
train_protocol_path_2="${protocol_path}/ASVSpoofData_2019_train_10_percent_laundered_protocol.txt"

echo -e $train_protocol_path_1 
echo -e $train_protocol_path_2 

grep LA_T $train_protocol_path_1 $train_protocol_path_2 | awk '{print $2}' > scp/train.lst

# validation set
val_protocol_path="${protocol_path}/ASVspoof2019.LA.cm.dev.trl.txt"

echo -e $val_protocol_path  
grep LA_D $val_protocol_path | awk '{print $2}' > scp/val.lst

# Combine protocol files to create a single protocol file 
awk -v OFS='\t' '{print $1, $2, $3, $4, $5}' $train_protocol_path_1 $train_protocol_path_2 $val_protocol_path > protocol.txt

# # evaluation set
# eval_protocol_path="$/ASVspoof2019.LA.cm.dev.trl.txt"
# grep LA_E protocol.txt | awk '{print $2}' > scp/test.lst
