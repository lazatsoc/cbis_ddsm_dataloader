from ddsm_dataset_factory import CBISDDSMDatasetFactory

dataset = CBISDDSMDatasetFactory('./config.json') \
        .drop_attributes("assessment", "breast_density", "subtlety") \
        .map_attribute_value('pathology', {'BENIGN_WITHOUT_CALLBACK': 'BENIGN'}) \
        .show_counts() \
        .create_classification('pathology')
dataset.visualize()