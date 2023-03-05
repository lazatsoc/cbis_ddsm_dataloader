from ddsm_dataset_factory import CBISDDSMDatasetFactory

dataset = CBISDDSMDatasetFactory('./config.json') \
        .train() \
        .add_masses() \
        .drop_attributes("assessment", "breast_density", "subtlety") \
        .map_attribute_value('pathology', {'BENIGN_WITHOUT_CALLBACK': 'BENIGN'}) \
        .show_counts() \
        .lesion_patches_random() \
        .create_classification('pathology')
dataset.visualize()