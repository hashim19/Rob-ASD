from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import io
import os
import pickle

CLIENT_SECRET_FILE = "client_secret.json"
SERVICE_ACCOUNT_FILE = 'robustness-eval-e528fd134c05.json'
API_NAME = "drive"
API_VERSION = "v3"
SCOPES = ["https://www.googleapis.com/auth/drive"]

# # Define the scopes
# SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Obtain your Google credentials
# def get_credentials():

#     pickle_file = f"token_{API_SERVICE_NAME}_{API_VERSION}.pickle"
#     # pickle_file = "token_drive_v3.pickle"

#     if os.path.exists(pickle_file):
#         with open(pickle_file, "rb") as token:
#             cred = pickle.load(token)

#     if not cred or not cred.valid:
#         if cred and cred.expired and cred.refresh_token:
#             cred.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
#             cred = flow.run_local_server()

#         with open(pickle_file, "wb") as token:
#             pickle.dump(cred, token)

#     # flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
#     # creds = flow.run_local_server(port=0)
#     return creds

# # Build the downloader
# creds = get_credentials()
# drive_downloader = build('drive', 'v3', credentials=creds)

# Create a Google Drive API service
def Create_Service(client_secret_file, api_name, api_version, *scopes):
    print(client_secret_file, api_name, api_version, scopes, sep="-")
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = [scope for scope in scopes[0]]
    print(SCOPES)

    cred = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    # cred = None

    # pickle_file = f"token_{API_SERVICE_NAME}_{API_VERSION}.pickle"

    # if os.path.exists(pickle_file):
    #     with open(pickle_file, "rb") as token:
    #         cred = pickle.load(token)

    # if not cred or not cred.valid:
    #     if cred and cred.expired and cred.refresh_token:
    #         cred.refresh(Request())
    #     else:
    #         flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
    #         cred = flow.run_local_server()

    #     with open(pickle_file, "wb") as token:
    #         pickle.dump(cred, token)

    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        print(API_SERVICE_NAME.capitalize(), "service created successfully.\n")
        return service
    except Exception as e:
        print("Unable to connect.")
        print(e)
        return None


# Create the service
service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

# Replace 'FOLDER_ID' with your actual Google Drive folder ID
# https://drive.google.com/drive/folders/1-c-nOBTdB9PK0QBNZrTT64jmK2fLCgbZ?usp=drive_link
folder_id = '1-c-nOBTdB9PK0QBNZrTT64jmK2fLCgbZ'
folder_name = 'RT_0.3'
# query = f"'{folder_id}'"
# print(query)
# results = service.files().list(q=query, pageSize=1000).execute()

out_folder = folder_name

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

page_token = None
# page_token = '1DXe01QIpLwHx7itnvbAiuG8wup_K1_t-'

total_num_files = 0
while True:
    results = service.files().list(
                q=f"'{folder_id}' in parents",
                pageSize=1000,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType)',
                pageToken=page_token
            ).execute()

    # print(results)

    print(page_token)

    # break

    items = results.get('files', [])

    # Download the files
    item_id = 0
    for item in items:
        print("Downloading Item {}".format(item_id))
        request = service.files().get_media(fileId=item['id'])

        out_file_path = os.path.join(out_folder, item['name'])

        if not os.path.exists(out_file_path):

            f = io.FileIO(out_file_path, 'wb')

            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}.")
        
        item_id += 1

    # print(f"Downloaded {len(items)} files from the folder.")

    total_num_files += item_id

    print("Total files downloaded = {}".format(total_num_files))

    page_token = results.get('nextPageToken', None)
    if page_token is None:
        break