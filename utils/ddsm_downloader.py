import concurrent.futures
import requests
import zipfile
from io import BytesIO
import os
import json
from tqdm import tqdm
import argparse

BASE_URL_IMAGE = 'https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={}'
BASE_URL_METADATA = 'https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeriesMetaData?SeriesInstanceUID={}'


class CBISDDSMDownloader:
    def __init__(self, manifest_path, download_path, skip_existing=True):
        self.__skip_existing = skip_existing
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

    def __exists(self, download_path, num_imgs):
        if os.path.exists(download_path):
            folder_contents = os.listdir(download_path)
            num_dcm = len(list(item for item in folder_contents if item.endswith('.dcm')))
            if num_dcm == num_imgs:
                return True
        return False

    def __download_extract_image(self, series_uid, path):
        response = requests.get(BASE_URL_IMAGE.format(series_uid))
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(path)

    def __payload(self, seriesUID):
        metadata = self.__get_metadata(seriesUID)
        folder_name = metadata['Subject ID']
        num_imgs = int(metadata['Number of Images'])
        download_path = os.path.join(self.__download_path, folder_name)

        if self.__skip_existing and self.__exists(download_path, num_imgs):
            return

        self.__download_extract_image(seriesUID, download_path)

    def start(self):
        self.__parse_manifest()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            future_to_url = {executor.submit(self.__payload, uid): uid for uid in self.__image_series_UID}
            for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(self.__image_series_UID), unit="file"):
                url = future_to_url[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"{url} generated an exception: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='CBIS DDSM Downloader')
    parser.add_argument('-m', '--manifest', default='../resources/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia',
                        help='Path to the manifest file.')
    parser.add_argument('-p', '--path', default='../CBIS_DDSM',
                        help='Path to the download folder. It will be created if not existing.')
    args = parser.parse_args()
    downloader = CBISDDSMDownloader(args.manifest, args.path)
    downloader.start()
