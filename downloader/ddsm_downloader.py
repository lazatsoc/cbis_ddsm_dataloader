import concurrent.futures
import requests
import zipfile
from io import BytesIO
import os
import json

BASE_URL_IMAGE = 'https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={}'
BASE_URL_METADATA = 'https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeriesMetaData?SeriesInstanceUID={}'

class CBISDDSMDownloader:
    def __init__(self, manifest_path, download_path='.', prefix_path=''):
        self.__prefix_path = prefix_path
        self.__download_path = download_path
        self.__manifest_file_path = manifest_path
        self.__image_series_UID = []
    
    def __parse_manifest(self):
        with open(self.__manifest_file_path) as file:
            found_starting_line_flag = False
            for line in file:
                if not found_starting_line_flag:
                    if 'ListOfSeriesToDownload=' in line:
                        found_starting_line_flag = True
                else:
                    self.__image_series_UID.append(line.strip())
            
            if found_starting_line_flag:
                print("Found {} items to download.".format(len(self.__image_series_UID)))
            else:
                print("Incorrect format of the manifest file provided!")
                
    
    def __get_metadata(self, series_uid):
        response = requests.get(BASE_URL_METADATA.format(series_uid))
        response_dict = json.loads(response.content.decode("utf-8"))[0]
        return response_dict


    def __download_extract_image(self, series_uid, path):
        response = requests.get(BASE_URL_IMAGE.format(series_uid))
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(path)

    def start(self):
        self.__parse_manifest()
        metadata = self.__get_metadata(self.__image_series_UID[0])
        folder_name = metadata['Subject ID']
        self.__download_extract_image(self.__image_series_UID[0], os.path.join(self.__download_path, folder_name))
        pass

if __name__ == "__main__":
    downloader = CBISDDSMDownloader('manifest/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia', download_path='CBIS_DDSM')
    downloader.start()