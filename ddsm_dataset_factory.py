import json
import os.path
import pandas as pd


class CBISDDSMDatasetFactory:
    def __init__(self, config_path) -> None:
        self.__config = self.__read_config(config_path)
        self.__masses = False
        self.__calcifications = False
        self.__train = False
        self.__test = False
        self.__total_df = None
        self.__actual_df = None

    @staticmethod
    def __read_config(config_path):
        with open(config_path, 'r') as cf:
            config = json.load(cf)
        return config

    def __read_total_df(self):
        csv_file_list = []
        if self.__train:
            csv_file_list.append(os.path.join(self.__config['download_path'], 'lesions_train.csv'))
        if self.__test:
            csv_file_list.append(os.path.join(self.__config['download_path'], 'lesions_test.csv'))

        self.__total_df = pd.concat((pd.read_csv(f) for f in csv_file_list), ignore_index=True)

    def add_masses(self):
        self.__masses = True
        return self

    def add_calcifications(self):
        self.__calcifications = True
        return self

    def train(self):
        self.__train = True
        return self

    def test(self):
        self.__test = True
        return self

    def show_statistics(self):
        self.__read_total_df()
        df = self.__total_df
        print(df["lesion_type"].value_counts(sort=True, ascending=False))
        print(df["type1"].value_counts(sort=True, ascending=False))
        print(df["type2"].value_counts(sort=True, ascending=False))
        print(df["pathology"].value_counts(sort=True, ascending=False))
        print(df["assessment"].value_counts(sort=True, ascending=False))
        print(df["breast_density"].value_counts(sort=True, ascending=False))
        print(df["subtlety"].value_counts(sort=True, ascending=False))
        return self


if __name__ == "__main__":
    factory = CBISDDSMDatasetFactory('./config.json') \
        .train() \
        .add_masses() \
        .show_statistics()
