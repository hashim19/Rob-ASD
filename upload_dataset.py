import os
import pandas as pd
from huggingface_hub import HfApi
from datasets import load_dataset, Dataset, Audio

api = HfApi()

# upload_folder(
#     folder_path="local/checkpoints",
#     repo_id="username/my-dataset",
#     repo_type="dataset",
#     multi_commits=True,
#     multi_commits_verbose=True,
# )

data_root_path = '/data/Data/ASVSpoofLaunderedDatabase/ASVspoofLD'
laundering_type = 'Reverberation'

metadata_filename = 'ASVspoofLauneredDatabase_' + laundering_type + '.txt'

audio_data_path = os.path.join(data_root_path, laundering_type, 'flac')
metadata_file_path = os.path.join(data_root_path, laundering_type, 'metadata.csv')

# metadata_file_path = os.path.join(data_root_path, 'protocols', metadata_filename)
# audio_data_path = os.path.join(data_root_path, laundering_type)


# api.upload_large_folder(
#     repo_id="hashim19/ASVspoofLD",
#     repo_type="dataset",
#     folder_path=data_path,
# )

####### Generating Metadata #########

# metadata = pd.read_csv(metadata_file_path, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])

# print(metadata)

# metadata = metadata.rename(columns={'AUDIO_FILE_NAME': 'file_name'}) 

# metadata['file_name'] = laundering_type + '/' + 'flac/' + metadata['file_name'].astype(str) + '.flac'

# print(metadata)

# metadata_path = os.path.join(audio_data_path, 'metadata.csv')
# metadata.to_csv(metadata_path, index=False)

############ Uploading the data ###########
metadata = pd.read_csv(metadata_file_path)

print(metadata)

audio_filepaths = metadata["file_name"].to_list()

# print(audio_filepaths)

# dataset = load_dataset("hashim19/ASVspoofLD")

# print(dataset)

audio_dataset = Dataset.from_dict({"audio": audio_filepaths}).cast_column("audio", Audio())

print(audio_dataset[0]["audio"])