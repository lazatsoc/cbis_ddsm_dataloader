import json
import argparse

from utils.ddsm_downloader import CBISDDSMDownloader
from utils.ddsm_png_converter import CBISDDSMConverter
from utils.ddsm_preprocessor import CBISDDSMPreprocessor

parser = argparse.ArgumentParser(prog='CBIS DDSM Preprocessor',
                                 description="Welcome to CBIS DDSM Dataloader library.\n"
                                             "Setup includes: \n"
                                             "    1. downloading the database, \n"
                                             "    2. converting images to PNG \n"
                                             "    3. organizing lesion information for PyTorch dataset creation \n"
                                             "Please ensure that the information in 'config.json' file is correctly set and the files exist.\n"
                                             "This process will take a while.\n")
parser.add_argument('-c', '--config_file', default='./config.json',
                    help='Path to the configuration file. Default=config.json')
args = parser.parse_args()

with open(args.config_file, 'r') as cf:
    config = json.load(cf)

# downloader = CBISDDSMDownloader(config['manifest'], config['download_path'])
# downloader.start()
#
# converter = CBISDDSMConverter(config['download_path'])
# converter.start()

preprocessor = CBISDDSMPreprocessor(config['download_path'],
                                    (config['mass_train_csv'], config['calc_train_csv']),
                                    (config['mass_test_csv'], config['calc_test_csv']))
preprocessor.start()
pass
