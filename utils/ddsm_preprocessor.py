import csv
import argparse
import os

import cv2
from PIL import Image
import numpy as np
import pandas
from matplotlib import pyplot as plt
from tqdm import tqdm
import concurrent.futures


class CBISDDSMPreprocessor:
    def __init__(self, download_path, csv_files_train, csv_files_test):
        self.__download_path = download_path
        self.__csv_files_train = csv_files_train
        self.__csv_files_test = csv_files_test
        self.__not_found = 0
        self.__other_errors = 0

    @staticmethod
    def __locate_lesion(mask_img, item_dict):
        mask = np.array(mask_img)
        ys, xs = np.where(mask)
        item_dict['minx'] = xs.min().tolist()
        item_dict['maxx'] = xs.max().tolist()
        item_dict['miny'] = ys.min().tolist()
        item_dict['maxy'] = ys.max().tolist()
        item_dict['cx'] = int((item_dict['maxx'] - item_dict['minx']) / 2 + item_dict['minx'])
        item_dict['cy'] = int((item_dict['maxy'] - item_dict['miny']) / 2 + item_dict['miny'])

        return True

    @staticmethod
    def __locate_breast(image, item_dict):
        image = (np.array(image) / 255).astype(np.uint8)
        threshold = int(0.05 * image.max())
        _, image_binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_areas = [cv2.contourArea(cont) for cont in contours]
        biggest_contour_idx = np.argmax(contours_areas)
        breast_contour = contours[biggest_contour_idx]
        breast_contour = cv2.convexHull(breast_contour)

        epsilon = 0.001 * cv2.arcLength(breast_contour, True)
        approxCurve = cv2.approxPolyDP(breast_contour, epsilon, True)

        # print(approxCurve.shape[0])
        # image_binary = np.dstack([image_binary]*3)
        # cv2.drawContours(image_binary, [approxCurve], 0, (255, 0, 0), 3)
        # plt.imshow(image_binary, cmap='gray')
        # plt.show()

        item_dict['breast_minx'] = breast_contour[:, 0, 0].min()
        item_dict['breast_maxx'] = breast_contour[:, 0, 0].max()
        item_dict['breast_miny'] = breast_contour[:, 0, 1].min()
        item_dict['breast_maxy'] = breast_contour[:, 0, 1].max()
        item_dict['breast_cx'] = int((item_dict['breast_maxx'] - item_dict['breast_minx']) / 2 + item_dict['breast_minx'])
        item_dict['breast_cy'] = int((item_dict['breast_maxy'] - item_dict['breast_miny']) / 2 + item_dict['breast_miny'])
        item_dict['breast_poly'] = approxCurve[:, 0, :].tolist()
        return True

    def __payload(self, row):
        item_dict = {
            "patient_id": row[0],
            "breast_density": row[1],
            "left_right": row[2],
            "view": row[3],
            "lesion_type": row[5],
            "type1": row[6],
            "type2": row[7],
            "assessment": row[8],
            "pathology": row[9],
            "subtlety": row[10],
            "image_path": os.path.splitext(row[11])[0] + '.png',
            "patch_path": os.path.splitext(row[12])[0] + '.png',
            "mask_path": os.path.splitext(row[13])[0] + '.png'
        }
        image = Image.open(os.path.join(self.__download_path, item_dict['image_path']))
        mask_img = Image.open(os.path.join(self.__download_path, item_dict['mask_path']))
        # CBIS-DDSM has the problem that sometimes the paths of the patch and the mask are swapped, so that
        # 'patch_path' = <path of the mask> and vice versa.
        # This is detected by comparing the image size and the mask size.
        # However, sometimes the mask image is a little smaller than the image regardless of whether they are swapped.
        # So we need to check if the mask is a lot smaller as well.
        if image.size != mask_img.size and (mask_img.size[0] < image.size[0] * 0.5 or mask_img.size[1] < image.size[1] * 0.5):
            tmp = item_dict['mask_path']
            item_dict['mask_path'] = item_dict['patch_path']
            item_dict['patch_path'] = tmp
            mask_img = Image.open(os.path.join(self.__download_path, item_dict['mask_path']))

        result = self.__locate_lesion(mask_img, item_dict)

        if not result:
            raise Exception()

        result = self.__locate_breast(image, item_dict)

        if not result:
            raise Exception()

        return item_dict

    def __parse_file(self, file_list, out_csv_path):
        rows = []
        data = []
        for csv_file in file_list:
            with open(csv_file) as fin:
                reader = csv.reader(fin, delimiter=',', quotechar='"')
                next(reader)

                for row in reader:
                    rows.append(row)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_row = {executor.submit(self.__payload, row): row for row in rows}
            for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(rows), unit='abnormalities'):
                try:
                    item_dict = future.result()
                except FileNotFoundError:
                    self.__not_found += 1
                    continue
                except Exception as e:
                    self.__other_errors += 1
                    continue

                data.append(item_dict)

        df = pandas.DataFrame(data)
        df.to_csv(out_csv_path)

    def start(self):
        print('Processing {} abnormality csv files for training.'.format(len(self.__csv_files_train)))
        self.__parse_file(self.__csv_files_train, os.path.join(self.__download_path, 'lesions_train.csv'))

        print('Processing {} abnormality csv files for testing.'.format(len(self.__csv_files_train)))
        self.__parse_file(self.__csv_files_test, os.path.join(self.__download_path, 'lesions_test.csv'))

        if self.__not_found > 0:
            print('Could not locate {} files. Please re-run the downloader.'.format(self.__not_found))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='CBIS DDSM Preprocessor')
    parser.add_argument('-p', '--path', default='../CBIS_DDSM',
                        help='Path to the download folder. It will be created if not existing.')
    parser.add_argument('-tr', '--csv_files_train', nargs='+',
                        default=['../resources/calc_case_description_train_set.csv',
                                 '../resources/mass_case_description_train_set.csv'],
                        help='One or more csv files to proces, as downloaded by TCIA repository.')
    parser.add_argument('-te', '--csv_files_test', nargs='+',
                        default=['../resources/calc_case_description_test_set.csv',
                                 '../resources/mass_case_description_test_set.csv'],
                        help='One or more csv files to proces, as downloaded by TCIA repository.')
    args = parser.parse_args()
    preprocessor = CBISDDSMPreprocessor(args.path, args.csv_files_train, args.csv_files_test)
    preprocessor.start()
