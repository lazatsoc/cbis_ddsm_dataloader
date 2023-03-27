import concurrent.futures
import os
import pydicom
from PIL import Image
from tqdm import tqdm
import argparse


class CBISDDSMConverter:
    def __init__(self, download_path, skip_existing=True, delete_dcm=False):
        self.__download_path = download_path
        self.__skip_existing = skip_existing
        self.__delete_dcm = delete_dcm
        self.__initialize_lists()

    def __initialize_lists(self):
        self.__dcm_image_list = []
        self.__to_delete_dcm_image_list = []
        self.__num_skipped = 0

    def __find_images(self, root_path):
        directory_list = os.listdir(root_path)
        for dir in directory_list:
            dir_path_1 = os.path.join(root_path, dir)
            if not os.path.isdir(dir_path_1):
                continue
            directory_list_1 = os.listdir(dir_path_1)
            for dir_1 in directory_list_1:
                dir_path_2 = os.path.join(dir_path_1, dir_1)
                directory_list_2 = os.listdir(dir_path_2)
                for dir_2 in directory_list_2:
                    dir_path = os.path.join(dir_path_2, dir_2)
                    contents_list = os.listdir(dir_path)
                    dcm_list = list(item for item in contents_list if item.endswith('.dcm'))
                    png_list = list(item for item in contents_list if item.endswith('.png'))
                    if self.__delete_dcm:
                        for img in dcm_list:
                            img_path = os.path.join(dir_path, img)
                            self.__to_delete_dcm_image_list.append(img_path)
                    if self.__skip_existing and len(dcm_list) == len(png_list):
                        self.__num_skipped += len(dcm_list)
                        continue
                    for img in dcm_list:
                        img_path = os.path.join(dir_path, img)
                        self.__dcm_image_list.append(img_path)
        print("Found {} dcm images to convert. Skipped {}.".format(len(self.__dcm_image_list), self.__num_skipped))

    @staticmethod
    def __get_png_path(dcm_path):
        path, name_ext = os.path.split(dcm_path)
        name, _ = os.path.splitext(name_ext)
        name_int = int(name) - 1  # Start numbering from 0
        name = str(name_int).zfill(6)
        output_path = os.path.join(path, name + '.png')
        return output_path

    @staticmethod
    def __dicom_to_png(input_path, output_path):
        ds = pydicom.dcmread(input_path, force=True)
        pixel_array = ds.pixel_array
        image = Image.fromarray(pixel_array)
        image.save(output_path, format='PNG', lossless=True)

    def __payload_convert(self, input_path):
        output_path = self.__get_png_path(input_path)
        self.__dicom_to_png(input_path, output_path)

    def __payload_delete(self, input_path):
        os.remove(input_path)

    def start(self):
        self.__initialize_lists()
        self.__find_images(self.__download_path)
        num_fails = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            future_to_url = {executor.submit(self.__payload_convert, uid): uid for uid in self.__dcm_image_list}
            for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(self.__dcm_image_list),
                               unit="file"):
                url = future_to_url[future]
                try:
                    future.result()
                except Exception as exc:
                    num_fails += 1
                    print(f"{url} generated an exception: {exc}")
            if num_fails > 0:
                print(
                    'Conversion failed for {} dcm images. Please re-run the downloader to fix incorrect downloads.'.format(
                        num_fails))
        if self.__delete_dcm:
            print('Cleaning up DICOM images...')
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Start the load operations and mark each future with its URL
                future_to_url = {executor.submit(self.__payload_delete, uid): uid for uid in self.__to_delete_dcm_image_list}
                for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(self.__to_delete_dcm_image_list),
                                   unit="file"):
                    url = future_to_url[future]
                    try:
                        future.result()
                    except Exception as exc:
                        num_fails += 1
                        print(f"{url} generated an exception: {exc}")
                if num_fails > 0:
                    print(
                        'Deletion failed for {} dcm images. Please re-run the downloader to fix incorrect downloads.'.format(
                            num_fails))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='CBIS DDSM Converter')
    parser.add_argument('-p', '--path', default='../CBIS_DDSM',
                        help='Path to the download folder. It will be created if not existing.')
    args = parser.parse_args()
    downloader = CBISDDSMConverter(args.path, delete_dcm=True)
    downloader.start()
