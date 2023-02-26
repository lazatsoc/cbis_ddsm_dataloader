import json
import os.path
import pandas as pd
from typing import List, Dict, Tuple

from datasets.ddsm_dataset import CBISDDSMCenteredPatchesDataset


class CBISDDSMDatasetFactory:
    def __init__(self, config_path) -> None:
        self.__config = self.__read_config(config_path)
        self.__train: bool = False
        self.__test: bool = False
        self.__dataframe = None
        self.__excluded_attrs: List[str] = []
        self.__excluded_values: Dict[str, set] = {'lesion_type': {'mass', 'calcification'}}
        self.__attribute_mapped_values: Dict[str, Dict[str, str]] = {}
        # self.__patch_transform

    @staticmethod
    def __read_config(config_path):
        with open(config_path, 'r') as cf:
            config = json.load(cf)
        return config

    def __fetch_filter_lesions(self):
        try:
            csv_file_list = []
            if self.__train:
                csv_file_list.append(os.path.join(self.__config['download_path'], 'lesions_train.csv'))
            if self.__test:
                csv_file_list.append(os.path.join(self.__config['download_path'], 'lesions_test.csv'))

            self.__dataframe = pd.concat((pd.read_csv(f) for f in csv_file_list), ignore_index=True)
        except:
            print('Database seems not properly set up. Please (re)run setup.py.')

        self.__dataframe.drop(self.__excluded_attrs, axis=1, inplace=True)

        for attribute, value_set in self.__excluded_values.items():
            self.__dataframe = self.__dataframe.loc[~self.__dataframe[attribute].isin(list(value_set))]

        for attribute, mapping in self.__attribute_mapped_values.items():
            for v1, v2 in mapping.items():
                self.__dataframe[attribute].replace(v1, v2, inplace=True)

    def add_masses(self):
        self.__excluded_values['lesion_type'].remove('mass')
        return self

    def add_calcifications(self):
        self.__excluded_values['lesion_type'].remove('calcification')
        return self

    def train(self):
        self.__train = True
        return self

    def test(self):
        self.__test = True
        return self

    def drop_attribute_values(self, attribute: str, *value_list: str):
        value_set = self.__excluded_values.get(attribute, set())
        for v in value_list:
            value_set.add(v)
        return self

    def map_attribute_value(self, attribute: str, mapping: Dict[str, str]):
        attribute_mapping = self.__attribute_mapped_values.get(attribute, dict())
        attribute_mapping.update(mapping)
        self.__attribute_mapped_values[attribute] = attribute_mapping
        return self

    def drop_attributes(self, *attribute_list: str):
        self.__excluded_attrs.extend(attribute_list)
        return self

    def show_counts(self):
        self.__fetch_filter_lesions()
        df = self.__dataframe
        labels_list = ["lesion_type", "type1", "type2", "pathology", "assessment", "breast_density", "subtlety"]
        print(os.linesep)
        for label in labels_list:
            if label in df.columns:
                print(df[label].value_counts(sort=True, ascending=False))
                print(os.linesep)

        return self

    def patches_centered(self, shape: Tuple[int] = (1024, 1024)):
        pass

    def create_classification(self, attribute):
        self.__fetch_filter_lesions()
        label_list = self.__dataframe[attribute].unique().tolist()
        return CBISDDSMCenteredPatchesDataset(self.__dataframe, self.__config['download_path'], attribute, label_list)


if __name__ == "__main__":
    dataset = CBISDDSMDatasetFactory('./config.json') \
        .train() \
        .add_masses() \
        .drop_attributes("assessment", "breast_density", "subtlety") \
        .map_attribute_value('pathology', {'BENIGN_WITHOUT_CALLBACK': 'BENIGN'}) \
        .show_counts() \
        .create_classification('pathology')
    dataset.visualize()
    pass