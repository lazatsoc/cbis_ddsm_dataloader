import csv
import argparse
import os
from PIL import Image
import pandas


class CBISDDSMPreprocessor:
    def __init__(self, download_path, csv_files_train, csv_files_test):
        self.__download_path = download_path
        self.__csv_files_train = csv_files_train
        self.__csv_files_test = csv_files_test

    def __locate_lesion(self, item_dict):
        image = Image.open(os.path.join(self.__download_path, item_dict['image_path']))
        mask = Image.open(os.path.join(self.__download_path, item_dict['mask_path']))
        pass

    def start(self):
        print('Processing {} training abnormality csv files.'.format(len(self.__csv_files_train)))
        columns = ["patient_id", "breast_density", "left_right", "view", "lesion_type", "type1", "type2", "assessment",
                   "pathology", "subtlety", "image_path", "roi_path"]
        for csv_file in self.__csv_files_train:
            # df = pandas.DataFrame(columns=columns)
            data = []
            with open(csv_file) as fin:
                reader = csv.reader(fin, delimiter=',', quotechar='|')
                next(reader)
                for row in reader:
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
                        "mask_path": os.path.splitext(row[12])[0] + '.png'
                    }
                    pos_dict = self.__locate_lesion(item_dict)
                    item_dict += pos_dict

                    break
        print('Processing {} testing abnormality csv files.'.format(len(self.__csv_files_test)))


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
