import concurrent.futures
import os
import pydicom
from PIL import Image
from tqdm import tqdm
import argparse


class CBISDDSMConverter:
    def __init__(self, download_path, skip_existing=True):
        self.__download_path = download_path
        self.__skip_existing = skip_existing
        self.__dcm_image_list = []
        self.__num_skipped = 0

    def __find_images(self, root_path):
        directory_list = os.listdir(root_path)
        for dir in directory_list:
            dir_path = os.path.join(root_path, dir)
            contents_list = os.listdir(dir_path)
            dcm_list = list(item for item in contents_list if item.endswith('.dcm'))
            png_list = list(item for item in contents_list if item.endswith('.png'))
            if self.__skip_existing and len(dcm_list) == len(png_list):
                self.__num_skipped += len(dcm_list)
                continue
            for img in dcm_list:
                img_path = os.path.join(dir_path, img)
                self.__dcm_image_list.append(img_path)
        print("Skipped {} dcm images.".format(self.__num_skipped))
        print("Found {} dcm images to convert.".format(len(self.__dcm_image_list)))

    def __dicom_to_png(self, input_path, output_path):
        ds = pydicom.dcmread(input_path, force=True)
        pixel_array = ds.pixel_array
        image = Image.fromarray(pixel_array)
        image.save(output_path, format='PNG', lossless=True)

    def __payload(self, input_path):
        output_path = input_path[:-4] + '.png'
        self.__dicom_to_png(input_path, output_path)

    def start(self):
        self.__find_images(self.__download_path)
        num_fails = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            future_to_url = {executor.submit(self.__payload, uid): uid for uid in self.__dcm_image_list}
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='CBIS DDSM Converter')
    parser.add_argument('-p', '--path', default='../CBIS_DDSM',
                        help='Path to the download folder. It will be created if not existing.')
    args = parser.parse_args()
    downloader = CBISDDSMConverter(args.path)
    downloader.start()
